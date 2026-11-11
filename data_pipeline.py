import copy
import gc
import json
from pathlib import Path

from common import DATA_REPO, DATA_REVISION, TEXT_MODEL, lock, sha256, task_id, write_json

GRAPH_PROTOCOL = 'strict-before-seconds-v2-earliest-train-statistics'


def download(data_root, dataset, task, *, prediction=False):
    from huggingface_hub import hf_hub_download
    import yaml
    task_id(dataset, task)
    data_root = Path(data_root)

    def fetch(name):
        return Path(hf_hub_download(DATA_REPO, repo_type='dataset', revision=DATA_REVISION,
                                   filename=f'{dataset}/{name}', local_dir=data_root))

    with lock(data_root / '.locks' / (dataset + '.lock')):
        manifest = yaml.safe_load(fetch('manifest.yaml').read_text())
        fetch(f'tasks/{task}/manifest.yaml')
        for split in ('train', 'val'):
            fetch(f'tasks/{task}/{split}.parquet')
        if prediction:
            fetch(f'tasks/{task}/test.parquet')
        else:
            for name in manifest['tables']:
                fetch(f'db/{name}.parquet')
    return data_root / dataset


def load_task(data_root, dataset, task):
    import relbench
    return relbench.load_dataset(Path(data_root) / dataset).load_task(task)


def seconds(values):
    import numpy as np
    import pandas as pd
    values = pd.to_datetime(values)
    values = values.fillna(pd.Timestamp('2262-01-01'))
    return values.to_numpy(dtype='datetime64[s]').astype(np.int64)


def graph_paths(cache_root, dataset, task):
    folder = Path(cache_root) / task_id(dataset, task)
    return folder / 'graph.pt', folder / 'graph.json'


def prepare_graph(args):
    import numpy as np
    import pandas as pd
    import torch
    import torch_frame as tf
    from torch_frame.config import TextEmbedderConfig
    from torch_frame.data import Dataset
    from torch_frame.utils import infer_df_stype
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import sort_edge_index

    path, meta_path = graph_paths(args.cache_root, args.dataset, args.task)
    with lock(path.parent / '.prepare.lock'):
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['protocol'] != GRAPH_PROTOCOL or meta['data_revision'] != DATA_REVISION:
                raise RuntimeError('Incompatible graph cache; use a fresh --cache_root.')
            return path
        download(args.data_root, args.dataset, args.task)
        task = load_task(args.data_root, args.dataset, args.task)
        train = task.get_table('train').df
        cutoff = pd.Timestamp(train[task.time_col].min())
        db = task.get_db(upto_test_timestamp=False)
        graph, stats, temporal_fields, audit = HeteroData(), {}, {}, {}
        embedder = None

        def embed(sentences):
            nonlocal embedder
            if embedder is None:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(TEXT_MODEL, revision=args.text_revision,
                                              device=args.prepare_device)
            return torch.from_numpy(embedder.encode(list(sentences), batch_size=256,
                                                   show_progress_bar=False))

        text_cfg = TextEmbedderConfig(text_embedder=embed, batch_size=256)
        for name, table in sorted(db.table_dict.items()):
            frame = table.df
            graph[name].num_nodes = len(frame)
            if table.pkey_col and not np.array_equal(frame[table.pkey_col].to_numpy(), np.arange(len(frame))):
                raise ValueError(f'{name}: official primary keys must equal row indices')
            prior = frame if table.time_col is None else frame.loc[frame[table.time_col] < cutoff]
            excluded = set(table.fkey_col_to_pkey_table) | {table.pkey_col}
            if len(prior):
                cols = infer_df_stype(prior.sample(min(1000, len(prior)), random_state=42))
                cols = {k: (tf.multicategorical if v == tf.embedding else v)
                        for k, v in cols.items() if k not in excluded}
            else:
                from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
                cols = {k: (tf.timestamp if is_datetime64_any_dtype(frame[k].dtype)
                            else tf.numerical if is_numeric_dtype(frame[k].dtype)
                            else tf.text_embedded) for k in frame if k not in excluded}
            if not cols:
                cols = {'__const__': tf.numerical}
                feature_frame = pd.DataFrame({'__const__': np.ones(len(frame))})
                reference = pd.DataFrame({'__const__': [1.]})
            elif len(prior):
                feature_frame = frame[list(cols)]
                reference = prior[list(cols)]
            else:
                feature_frame = frame[list(cols)].copy()
                synthetic = {}
                for column, kind in cols.items():
                    if kind == tf.numerical:
                        synthetic[column] = [0., 1.]
                    elif kind == tf.timestamp:
                        synthetic[column] = [cutoff - pd.Timedelta(days=366), cutoff - pd.Timedelta(days=1)]
                    else:
                        synthetic[column] = ['', '']
                        feature_frame[column] = feature_frame[column].map(str)
                reference = pd.DataFrame(synthetic)
            reference_ds = Dataset(reference, col_to_stype=cols,
                                   col_to_text_embedder_cfg=text_cfg).materialize()
            reference_stats = copy.deepcopy(reference_ds.col_stats)
            materialized = Dataset(feature_frame, col_to_stype=cols,
                                   col_to_text_embedder_cfg=text_cfg).materialize(col_stats=reference_stats)
            graph[name].tf = materialized.tensor_frame
            stats[name] = materialized.col_stats
            audit[name] = dict(rows=len(frame), statistics_rows=len(prior),
                               statistics_max_time=(str(prior[table.time_col].max())
                                                    if table.time_col and len(prior) else None),
                               features={k: str(v) for k, v in cols.items()})
            if table.time_col:
                graph[name].time = torch.from_numpy(seconds(frame[table.time_col]))
                fields = []
                for i, (column, target) in enumerate(sorted(table.fkey_col_to_pkey_table.items())):
                    field = f'temporal_entity_{i}'
                    graph[name][field] = torch.tensor(frame[column].fillna(-1).to_numpy(dtype=np.int64))
                    fields.append(field)
                temporal_fields[name] = fields
            for column, target in sorted(table.fkey_col_to_pkey_table.items()):
                mask = frame[column].notna().to_numpy()
                src = torch.from_numpy(np.flatnonzero(mask))
                dst = torch.from_numpy(frame.loc[mask, column].to_numpy(dtype=np.int64))
                if len(dst) and (int(dst.min()) < 0 or int(dst.max()) >= len(db.table_dict[target].df)):
                    raise ValueError(f'{name}.{column}: invalid foreign key')
                graph[name, 'f2p_' + column, target].edge_index = sort_edge_index(torch.stack([src, dst]))
                graph[target, 'rev_f2p_' + column, name].edge_index = sort_edge_index(torch.stack([dst, src]))
            print(f'PREPARE {name}: rows={len(frame)} prior-statistics={len(prior)}', flush=True)
            del reference_ds, materialized
            gc.collect()
        graph.validate(raise_on_error=True)
        meta = dict(protocol=GRAPH_PROTOCOL, data_repo=DATA_REPO, data_revision=DATA_REVISION,
                    dataset=args.dataset, task=args.task, statistics_cutoff=str(cutoff),
                    statistics_policy='strictly before earliest training seed; static tables are timeless',
                    hidden_columns=task.hidden_columns(), tables=audit,
                    text_model=TEXT_MODEL, text_revision=args.text_revision,
                    test_labels_loaded=False, temporal_fields=temporal_fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(data=graph, stats=stats, temporal_fields=temporal_fields), temporary)
        temporary.replace(path)
        meta['graph_sha256'] = sha256(path)
        write_json(meta_path, meta)
        return path


def read_graph(args):
    import torch
    path, meta_path = graph_paths(args.cache_root, args.dataset, args.task)
    if not path.exists():
        raise FileNotFoundError(f'Run --mode prepare first: {path}')
    meta = json.loads(meta_path.read_text())
    if meta['protocol'] != GRAPH_PROTOCOL or meta['data_revision'] != DATA_REVISION:
        raise RuntimeError('Graph protocol/revision mismatch')
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False), meta


class CausalTransform:
    def __init__(self, entity, times, targets=None):
        self.entity, self.times, self.targets = entity, times, targets

    def __call__(self, batch):
        import torch
        root = batch[self.entity]
        original_times = self.times[root.input_id]
        root.seed_time = original_times
        if self.targets is not None:
            root.y = self.targets[root.input_id]
        invalid_roots = torch.zeros(len(original_times), dtype=torch.bool)
        if 'time' in root:
            invalid_roots = root.time[:len(original_times)] >= original_times
        for name in batch.node_types:
            store = batch[name]
            groups = store.batch
            available = ~invalid_roots[groups]
            if 'time' in store:
                causal = store.time < original_times[groups]
                nonroot = causal.clone()
                if name == self.entity:
                    nonroot[:len(original_times)] = True
                if not bool((nonroot | ~available).all()):
                    raise RuntimeError(f'Temporal leakage detected by sampler audit: {name}')
                available &= causal
            store.available = available
            for key in list(store.keys()):
                if key.startswith('temporal_entity_'):
                    store[key] = store[key].clone()
                    store[key][~available] = -1
        for relation in batch.edge_types:
            src, _, dst = relation
            edges = batch[relation].edge_index
            keep = batch[src].available[edges[0]] & batch[dst].available[edges[1]]
            batch[relation].edge_index = edges[:, keep]
        return batch


def make_loader(data, task, frame, cfg, *, training=False, labels=True):
    import torch
    from torch_geometric.loader import NeighborLoader
    times = torch.from_numpy(seconds(frame[task.time_col]))
    targets = torch.tensor(frame[task.target_col].to_numpy(), dtype=torch.float) if labels else None
    return NeighborLoader(data,
                          input_nodes=(task.entity_table, torch.tensor(frame[task.entity_col].to_numpy(), dtype=torch.long)),
                          input_time=times - 1, time_attr='time', disjoint=True,
                          num_neighbors=[max(1, cfg.num_neighbors // (2 ** i)) for i in range(cfg.num_layers)],
                          batch_size=cfg.batch_size, shuffle=training, num_workers=0,
                          temporal_strategy=cfg.temporal_strategy, subgraph_type='directional',
                          transform=CausalTransform(task.entity_table, times, targets))
