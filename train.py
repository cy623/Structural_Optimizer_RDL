"""Validation-only model selection and separately frozen test prediction export."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

from common import DATA_REVISION, OFFICIAL_COMMIT, lock, reseed, sha256, source_hash, task_id, write_json
from data_pipeline import download, load_task, make_loader, read_graph

GATE_THRESHOLDS = [.5, .7, .78, .82, .86, .9]

def locations(args):
    name = task_id(args.dataset, args.task)
    root = Path(args.output_root)
    return name, root / 'best_model' / name / 'all', root / 'runtime' / 'locks' / (name + '.lock')


def make_model(graph, cfg, args, device):
    from model import Model
    return Model(args, graph['data'], graph['stats'], temporal_fields=graph['temporal_fields'],
                 config=vars(cfg)).to(device)


def validation(model, loader, entity, device, seed, thresholds):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    py, npstate, was_training = random.getstate(), np.random.get_state(), model.training
    result = {}
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if device.type == 'cuda' else []):
            model.eval()
            reseed(seed)
            same_masks = {}
            for threshold in thresholds:
                same_masks.setdefault(model.gate_signature(threshold), []).append(threshold)
            scores, labels = {threshold: [] for threshold in thresholds}, []
            for batch in loader:
                batch = batch.to(device)
                for group in same_masks.values():
                    model.cfg.gate_threshold = group[0]
                    logits, _ = model(batch, entity)
                    for threshold in group:
                        scores[threshold].append(logits.cpu())
                labels.append(batch[entity].y.cpu())
            model.cfg.gate_threshold = thresholds[-1]  # as left by one pass per threshold
            labels = torch.cat(labels).numpy()
            for threshold in thresholds:
                result[threshold] = float(roc_auc_score(labels, torch.cat(scores[threshold]).numpy()))
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        model.train(was_training)
    return result


def train(args, config):
    import numpy as np
    import torch
    name, folder, lock_path = locations(args)
    freeze = Path(args.output_root) / 'frozen' / (name + '.json')
    if freeze.exists():
        raise RuntimeError('Test predictions already frozen for this task. HPO after test export is disabled.')
    graph, graph_meta = read_graph(args)
    task = load_task(args.data_root, args.dataset, args.task)
    tables = {s: task.get_table(s).df for s in ('train', 'val')}
    cfg = SimpleNamespace(**config)
    device = torch.device('cpu' if args.gpu == 'cpu' else 'cuda:0')
    thresholds = thresholds = sorted(set(GATE_THRESHOLDS))
    trial = args.trial_id or hashlib.sha256(json.dumps([config, thresholds], sort_keys=True).encode()).hexdigest()[:16]
    rows_path = Path(args.output_root) / 'results' / (name + '_all_results.jsonl')
    identity = dict(source_hash=source_hash(), graph_sha256=graph_meta['graph_sha256'])
    with lock(lock_path):
        if rows_path.exists():
            for line in rows_path.read_text().splitlines():
                previous = json.loads(line)
                if previous['trial_id'] == trial:
                    if previous['config'] != config or any(previous[k] != v for k, v in identity.items()):
                        raise RuntimeError('Trial ID conflicts with an existing configuration/source/cache')
                    print('SKIP completed', trial, flush=True)
                    return previous
    reseed(cfg.seed)
    loaders = {s: make_loader(graph['data'], task, tables[s], cfg, training=s == 'train') for s in tables}
    model = make_model(graph, cfg, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best, snapshot, best_epoch, best_threshold = -float('inf'), None, None, None
    epoch_rows = []
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        model.anneal(epoch)
        total, count = 0., 0
        for step, batch in enumerate(loaders['train']):
            if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, terms = model(batch, task.entity_table)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch[task.entity_table].y)
            loss = loss + model.regularization(terms)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {name} {trial} epoch={epoch}')
            loss.backward()
            optimizer.step()
            count += len(logits)
            total += float(loss.detach()) * len(logits)
        if not count:
            raise RuntimeError('Empty training epoch')
        scores = validation(model, loaders['val'], task.entity_table, device, cfg.seed, thresholds)
        threshold = max(scores, key=scores.get)
        if scores[threshold] > best:
            best, best_epoch, best_threshold = scores[threshold], epoch, threshold
            snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_rows.append(dict(epoch=epoch, loss=total / count,
                               val_auc_by_threshold={str(k): v for k, v in scores.items()}))
        print(f'{name} {trial} epoch={epoch}/{cfg.epochs} loss={total/count:.6f} '
              f'val_auc={scores[threshold]:.8f} gate_threshold={threshold} best={best:.8f} '
              f'elapsed={time.time() - started:.0f}s', flush=True)
    selected_config = dict(config, gate_threshold=best_threshold)
    result = dict(trial_id=trial, dataset=args.dataset, task=args.task, config=config,
                  selected_config=selected_config, best_val_auc=best, best_epoch=best_epoch,
                  gate_thresholds=thresholds, seed=cfg.seed, training_seconds=time.time() - started,
                  epochs=epoch_rows, selection_metric='validation_roc_auc',
                  test_labels_loaded=False, test_evaluated=False,
                  official_commit=OFFICIAL_COMMIT, data_revision=DATA_REVISION, **identity)
    import importlib.metadata
    result['environment'] = {n: importlib.metadata.version(n) for n in
                             ['torch', 'pytorch-frame', 'torch-geometric', 'pyg-lib',
                              'torch-scatter', 'numpy', 'pandas', 'pyarrow']}
    result['environment']['cuda'] = torch.version.cuda
    with lock(lock_path):
        if freeze.exists():
            raise RuntimeError('Concurrent freeze: stop the HPO controller before predicting')
        params = folder / 'params.json'
        winner = json.loads(params.read_text()) if params.exists() else None
        if winner and any(winner[k] != v for k, v in identity.items()):
            raise RuntimeError('Do not mix code or graph versions in one output root')
        if winner is None or best > winner['best_val_auc']:
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / '.best.tmp'
            torch.save(dict(model=snapshot, config=selected_config, **identity), temporary)
            temporary.replace(folder / 'best.pt')
            write_json(params, dict(result, checkpoint_sha256=sha256(folder / 'best.pt')))
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open('a') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        txt_path = rows_path.with_suffix('.txt')
        with txt_path.open('a') as stream:
            stream.write('_'.join(f'{k}{v}' for k, v in selected_config.items()) + '\n')
            stream.write(json.dumps(dict(val_metric_mean=best, val_metric_std=0.,
                                         best_epoch=best_epoch, trial_id=trial,
                                         training_seconds=result['training_seconds'],
                                         test_evaluated=False), allow_nan=False) + '\n')
    return result


def predict(args):
    import numpy as np
    import pandas as pd
    import torch
    name, folder, lock_path = locations(args)
    freeze_path = Path(args.output_root) / 'frozen' / (name + '.json')
    with lock(lock_path):
        params = json.loads((folder / 'params.json').read_text())
        checkpoint = folder / 'best.pt'
        if params['checkpoint_sha256'] != sha256(checkpoint):
            raise RuntimeError('best.pt does not match the checkpoint_sha256 in params.json')
        freeze = dict(dataset=args.dataset, task=args.task, config=params['selected_config'],
                      best_val_auc=params['best_val_auc'], best_epoch=params['best_epoch'],
                      checkpoint_sha256=params['checkpoint_sha256'], source_hash=params['source_hash'],
                      graph_sha256=params['graph_sha256'], official_commit=OFFICIAL_COMMIT,
                      data_revision=DATA_REVISION, selection_metric='validation_roc_auc',
                      test_labels_loaded=False)
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
            raise RuntimeError('Existing freeze differs; prediction export refused')
        write_json(freeze_path, freeze)
    # The model/configuration are frozen BEFORE fetching test data.
    download(args.data_root, args.dataset, args.task, prediction=True)
    task = load_task(args.data_root, args.dataset, args.task)
    test_path = Path(args.data_root) / args.dataset / 'tasks' / args.task / 'test.parquet'
    keys = [task.entity_col, task.time_col]
    frame = pd.read_parquet(test_path, columns=keys)  # test TARGET column is never read
    if frame.duplicated(keys).any():
        raise ValueError('Official test keys contain duplicates')
    graph, meta = read_graph(args)
    device = torch.device('cpu' if args.gpu == 'cpu' else 'cuda:0')
    cfg = SimpleNamespace(**freeze['config'])
    reseed(cfg.seed)
    model = make_model(graph, cfg, args, device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    loader = make_loader(graph['data'], task, frame, cfg, labels=False)
    probabilities, seen = np.empty(len(frame)), np.zeros(len(frame), dtype=bool)
    with torch.no_grad():
        reseed(cfg.seed)
        for batch in loader:
            ids = batch[task.entity_table].input_id.numpy()
            if seen[ids].any():
                raise ValueError('Duplicate sampled test input IDs')
            seen[ids] = True
            logits, _ = model(batch.to(device), task.entity_table)
            probabilities[ids] = logits.sigmoid().cpu().numpy()
    if not seen.all() or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError('Missing or invalid prediction probabilities')
    frame[task.target_col] = probabilities
    output = Path(args.output_root) / 'predictions' / (name + '.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    write_json(Path(args.output_root) / 'prediction_reports' / (name + '.json'),
               dict(freeze, rows=len(frame), csv_sha256=sha256(output), test_labels_loaded=False))
    print(f'PREDICTIONS {output} rows={len(frame)} (no test labels scored)', flush=True)
    return output
