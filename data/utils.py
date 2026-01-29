from .database import Database
from .table import Table
from typing import Any, Dict, NamedTuple, Optional, Tuple, Iterator, Union, List
from torch_frame import stype
from torch_frame.utils import infer_df_stype
import torch
from torch_frame.config import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
import os
import numpy as np
import pandas as pd
from torch_geometric.utils import sort_edge_index
from .task_entity import EntityTask
from .task_base import TaskType
from torch_geometric.typing import NodeType
from torch import Tensor
from torch_geometric.loader import NeighborLoader
# from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.loader import NodeLoader
import random



def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp (in seconds)."""
    assert ser.dtype in [np.dtype("datetime64[s]"), np.dtype("datetime64[ns]")]
    unix_time = ser.astype("int64").values
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    return unix_time


def remove_pkey_fkey(col_to_stype: Dict[str, Any], table: Table) -> dict:
    r"""Remove pkey, fkey columns since they will not be used as input feature."""
    if table.pkey_col is not None:
        if table.pkey_col in col_to_stype:
            col_to_stype.pop(table.pkey_col)
    for fkey in table.fkey_col_to_pkey_table.keys():
        if fkey in col_to_stype:
            col_to_stype.pop(fkey)


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, stype]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.table_dict.items():
        df = table.df
        df = df.sample(min(1_000, len(df)))
        inferred_col_to_stype = infer_df_stype(df)
        # Hack for now. This is relevant for rel-amazon.
        for col, stype_ in inferred_col_to_stype.items():
            if stype_.value == "embedding":
                inferred_col_to_stype[col] = stype.multicategorical
        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype

    return inferred_col_to_stype_dict



def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData() # Create a new PyG heterogeneous graph container
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():  # Traverse every table in the database
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name] # Get the "column → feature type" mapping of the table

        # Remove pkey, fkey columns since they will not be used as input feature.
        remove_pkey_fkey(col_to_stype, table)

        # Boundary Processing
        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        # save path
        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        # Materialize DataFrame into Tensors using custom Dataset
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(         # If the table has a time column, it is converted to a Unix timestamp
                to_unix_time(table.df[table.time_col])        #  and attached to the .time attribute of the node type.
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():     
            # eg: dict_items([('raceId', 'races'), ('constructorId', 'constructors')])
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    # HeteroData(
    #     drivers={ tf=TensorFrame([857, 6]) },
    #     races={
    #         tf=TensorFrame([820, 5]),
    #         time=[820],
    #     },
    #     constructor_standings={
    #         tf=TensorFrame([10170, 4]),
    #         time=[10170],
    #     },
    #     constructor_results={
    #         tf=TensorFrame([9408, 2]),
    #         time=[9408],
    #     },
    #     results={
    #         tf=TensorFrame([20323, 11]),
    #         time=[20323],
    #     },
    #     qualifying={
    #         tf=TensorFrame([4082, 3]),
    #         time=[4082],
    #     },
    #     circuits={ tf=TensorFrame([77, 7]) },
    #     constructors={ tf=TensorFrame([211, 3]) },
    #     standings={
    #         tf=TensorFrame([28115, 4]),
    #         time=[28115],
    #     },
    #     (races, f2p_circuitId, circuits)={ edge_index=[2, 820] },
    #     (circuits, rev_f2p_circuitId, races)={ edge_index=[2, 820] },
    #     (constructor_standings, f2p_raceId, races)={ edge_index=[2, 10170] },
    #     (races, rev_f2p_raceId, constructor_standings)={ edge_index=[2, 10170] },
    #     (constructor_standings, f2p_constructorId, constructors)={ edge_index=[2, 10170] },
    #     (constructors, rev_f2p_constructorId, constructor_standings)={ edge_index=[2, 10170] },
    #     (constructor_results, f2p_raceId, races)={ edge_index=[2, 9408] },
    #     (races, rev_f2p_raceId, constructor_results)={ edge_index=[2, 9408] },
    #     (constructor_results, f2p_constructorId, constructors)={ edge_index=[2, 9408] },
    #     (constructors, rev_f2p_constructorId, constructor_results)={ edge_index=[2, 9408] },
    #     (results, f2p_raceId, races)={ edge_index=[2, 20323] },
    #     (races, rev_f2p_raceId, results)={ edge_index=[2, 20323] },
    #     (results, f2p_driverId, drivers)={ edge_index=[2, 20323] },
    #     (drivers, rev_f2p_driverId, results)={ edge_index=[2, 20323] },
    #     (results, f2p_constructorId, constructors)={ edge_index=[2, 20323] },
    #     (constructors, rev_f2p_constructorId, results)={ edge_index=[2, 20323] },
    #     (qualifying, f2p_raceId, races)={ edge_index=[2, 4082] },
    #     (races, rev_f2p_raceId, qualifying)={ edge_index=[2, 4082] },
    #     (qualifying, f2p_driverId, drivers)={ edge_index=[2, 4082] },
    #     (drivers, rev_f2p_driverId, qualifying)={ edge_index=[2, 4082] },
    #     (qualifying, f2p_constructorId, constructors)={ edge_index=[2, 4082] },
    #     (constructors, rev_f2p_constructorId, qualifying)={ edge_index=[2, 4082] },
    #     (standings, f2p_raceId, races)={ edge_index=[2, 28115] },
    #     (races, rev_f2p_raceId, standings)={ edge_index=[2, 28115] },
    #     (standings, f2p_driverId, drivers)={ edge_index=[2, 28115] },
    #     (drivers, rev_f2p_driverId, standings)={ edge_index=[2, 28115] }
    # )

    return data, col_stats_dict


class AttachTargetTransform:
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of disjoins subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class NodeTrainTableInput(NamedTuple):
    r"""Training table input for node prediction.

    - nodes is a Tensor of node indices.
    - time is a Tensor of node timestamps.
    - target is a Tensor of node labels.
    - transform attaches the target to the batch.
    """

    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(
    table: Table,
    task: EntityTask,
) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values) # get the training nodes index

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))  # get the training nodes timestemp

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None    
    if task.target_col in table.df:                              # get the training labels
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


def data_loader(args, train_table, val_table, test_table, data, task):
    loader_dict = {}

    for split, table in [
        ("train", train_table),   # training table
        ("val", val_table),
        ("test", test_table),
    ]:  
        
        # Get the training table input for node prediction.
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )
        # NodeTrainTableInput(nodes=('drivers', tensor([12, 20, 10,  ..., 77, 43, 56])), time=tensor([1091577600, 1091577600, 1088985600,  ...,  764985600,  762393600,
        #  762393600]), target=tensor([0., 0., 0.,  ..., 0., 0., 0.], dtype=torch.float64), transform=<data.utils.AttachTargetTransform object at 0x7f781b331b40>)
        entity_table = table_input.nodes[0]
        if args.model == 'RelGNN':
            loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],  # we sample subgraphs of depth 2, 128 neighbors per node.
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            subgraph_type=args.subgraph_type,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        else:
            loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],  # we sample subgraphs of depth 2, 128 neighbors per node.
                time_attr="time",
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=args.batch_size,
                temporal_strategy=args.temporal_strategy,
                shuffle=split == "train",
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
            )
    return entity_table, loader_dict


def get_configs(dataset, task):

    if dataset == 'rel-amazon' and task == 'user-churn':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 32,
        }
        loader_config = {
            'batch_size': 4096,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'item-churn':
        model_config = {
            'num_model_layers': 4,
            'channels': 256,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config      
    
    if dataset == 'rel-avito' and task == 'user-clicks':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'user-visits':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-repeat':
        model_config = {
            'num_model_layers': 1,
            'channels': 32,
            'aggr': 'sum',
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-ignore':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-f1' and task == 'driver-dnf':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-f1' and task == 'driver-top3':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 2,
            'simplified_MP': True,
        }
        loader_config = {
            'batch_size': 256,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-hm' and task == 'user-churn':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-engagement':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 1024,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-badge':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'study-outcome':
        model_config = {
            'num_model_layers': 2,
            'channels': 256,
            'aggr': 'sum',
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'user-ltv':
        model_config = {
            'num_model_layers': 2,
            'channels': 32,
            'aggr': 'sum',
            'num_heads': 64,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-amazon' and task == 'item-ltv':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'ad-ctr':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-event' and task == 'user-attendance':
        model_config = {
            'num_model_layers': 2,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-f1' and task == 'driver-position':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 512,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'bidirectional',
        }
        return model_config, loader_config

    if dataset == 'rel-hm' and task == 'item-sales':
        model_config = {
            'num_model_layers': 4,
            'channels': 128,
            'aggr': 'sum',
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 2048,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'post-votes':
        model_config = {
            'num_model_layers': 2,
            'channels': 64,
            'aggr': 'sum',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 1024,
            'num_neighbors': 128,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'study-adverse':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'mean',
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 64,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config

    if dataset == 'rel-trial' and task == 'site-success':
        model_config = {
            'num_model_layers': 1,
            'channels': 128,
            'aggr': 'mean',
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 128,
            'num_neighbors': 64,
            'num_layers': 2,
            'subgraph_type': 'directional',
        }
        return model_config, loader_config
    
    
    if dataset == 'rel-amazon' and task == 'user-item-purchase':
        model_config = {
            'num_heads': 2,
        }
        loader_config = {
            'batch_size': 4096,
        }
        return model_config, loader_config

    if dataset == 'rel-amazon' and task == 'user-item-rate':
        model_config = {
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
        }
        return model_config, loader_config

    if dataset == 'rel-amazon' and task == 'user-item-review':
        model_config = {
            'num_heads': 1,
        }
        loader_config = {
            'batch_size': 256,
        }
        return model_config, loader_config

    if dataset == 'rel-avito' and task == 'user-ad-visit':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 16,
        }
        loader_config = {
            'batch_size': 256,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-hm' and task == 'user-item-purchase':
        model_config = {
            'num_model_layers': 1,
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 2048,
            'num_layers': 2,
        }
        return model_config, loader_config
    
    if dataset == 'rel-stack' and task == 'user-post-comment':
        model_config = {
            'num_model_layers': 4,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-stack' and task == 'post-post-related':
        model_config = {
            'num_model_layers': 2,
            'num_heads': 8,
        }
        loader_config = {
            'batch_size': 512,
            'num_layers': 2,
        }
        return model_config, loader_config

    if dataset == 'rel-trial' and task == 'condition-sponsor-run':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 128,
            'num_layers': 4,
        }
        return model_config, loader_config
    
    if dataset == 'rel-trial' and task == 'site-sponsor-run':
        model_config = {
            'num_model_layers': 8,
            'num_heads': 4,
        }
        loader_config = {
            'batch_size': 64,
            'num_layers': 4,
        }
        return model_config, loader_config
