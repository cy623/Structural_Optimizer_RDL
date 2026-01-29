from .database import Database
from .table import Table
from typing import Any, Dict, NamedTuple, Optional, Tuple, Iterator, Union, List
from torch_frame import stype
from torch_frame.utils import infer_df_stype
import torch
from torch_frame.config import TextEmbedderConfig
from torch_frame.data.stats import StatType
# from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
import os
import numpy as np
import pandas as pd
from torch_geometric.utils import sort_edge_index
from .task_entity import EntityTask
from.task_recommendation import RecommendationTask
from .task_base import TaskType
from torch_geometric.typing import NodeType
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch.utils.data import DataLoader, Dataset, Sampler
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




class LinkTrainTableInput(NamedTuple):
    r"""Training table input for link prediction.

    - src_nodes is a Tensor of source node indices.
    - dst_nodes is PyTorch sparse tensor in csr format.
        dst_nodes[src_node_idx] gives a tensor of destination node
        indices for src_node_idx.
    - num_dst_nodes is the total number of destination nodes.
        (used to perform negative sampling).
    - src_time is a Tensor of time for src_nodes
    """

    src_nodes: Tuple[NodeType, Tensor]
    dst_nodes: Tuple[NodeType, Tensor]
    num_dst_nodes: int
    src_time: Optional[Tensor]


def get_link_train_table_input(
    table: Table,
    task: RecommendationTask,
) -> LinkTrainTableInput:
    r"""Get the training table input for link prediction."""

    src_node_idx: Tensor = torch.from_numpy(
        table.df[task.src_entity_col].astype(int).values
    )
    exploded = table.df[task.dst_entity_col].explode()
    coo_indices = torch.from_numpy(
        np.stack([exploded.index.values, exploded.values.astype(int)])
    )
    sparse_coo = torch.sparse_coo_tensor(
        coo_indices,
        torch.ones(coo_indices.size(1), dtype=bool),
        (len(src_node_idx), task.num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    return LinkTrainTableInput(
        src_nodes=(task.src_entity_table, src_node_idx),
        dst_nodes=(task.dst_entity_table, dst_node_indices),
        num_dst_nodes=task.num_dst_nodes,
        src_time=time,
    )



class CustomNodeLoader(NodeLoader):

    def get_neighbors(
        self,
        input_data: NodeSamplerInput,
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.node_sampler.sample_from_nodes(input_data)
        out = self.filter_fn(out)
        return out


class TimestampSampler(Sampler[int]):
    r"""A TimestampSampler that samples rows from the same timestamp."""

    def __init__(
        self,
        timestamp: Tensor,
        batch_size: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.time_dict = {
            int(time): (timestamp == time).nonzero().view(-1)
            for time in timestamp.unique()
        }
        self.num_batches = sum(
            [indices.numel() // batch_size for indices in self.time_dict.values()]
        )

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = []
        for indices in self.time_dict.values():
            # Random shuffle values:
            indices = indices[torch.randperm(indices.numel())]
            batches = torch.split(indices, self.batch_size)
            for batch in batches:
                if len(batch) < self.batch_size:
                    continue
                else:
                    all_batches.append(batch.tolist())

        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return self.num_batches



class CustomLinkDataset(Dataset):
    r"""A custom link prediction dataset.

    Sample source nodes, time, and one positive destination node.
    """

    def __init__(
        self,
        src_node_indices: Tensor,
        dst_node_indices: Tensor,  # CSR sparse matrix
        num_dst_nodes: int,
        src_time: Tensor,
    ):
        assert len(src_node_indices) == len(dst_node_indices) and len(
            src_node_indices
        ) == len(src_time)
        self.src_node_indices = src_node_indices
        self.dst_node_indices = dst_node_indices
        self.num_dst_nodes = num_dst_nodes
        self.src_time = src_time

    def __getitem__(self, index) -> Tensor:
        r"""Returns 1-dim tensor of size 3.

        - source node index
        - positive destination node index
        - source node time
        """
        return torch.tensor(
            [
                self.src_node_indices[index],
                random.choice(self.dst_node_indices[index].indices()[0]),
                self.src_time[index],
            ]
        )

    def __len__(self):
        return len(self.src_node_indices)





class LinkNeighborLoader(DataLoader):
    r"""A custom neighbor loader for link prediction.
    Based on https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html

    Args:
        src_nodes (Tuple[NodeType, Tensor]): A tensor of source node indices.
        dst_nodes (Tuple[NodeType, Tensor]): A csr sparse tensor, where
            dst_nodes[index] is a list of destination node indices
            for src_nodes[index] at src_time[index].
        num_dst_nodes (int): Total number of destination nodes. Used to
            determine the range of negative samples.
        src_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        share_same_time (bool): Whether to share the seed time within mini-batch
            or not (default: :obj:`False`)
    """

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        src_nodes: Tuple[NodeType, Tensor],
        dst_nodes: Tuple[NodeType, Tensor],
        num_dst_nodes: int,
        src_time: OptTensor = None,
        share_same_time: bool = False,
        subgraph_type: Union[SubgraphType, str] = "directional",
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        **kwargs,
    ):
        node_sampler = NeighborSampler(
            data,
            num_neighbors=num_neighbors,
            subgraph_type=subgraph_type,
            disjoint=True,
            temporal_strategy=temporal_strategy,
            time_attr=time_attr,
            share_memory=kwargs.get("num_workers", 0) > 0,
        )

        self.data = data
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes
        self.num_dst_nodes = num_dst_nodes
        self.src_time = src_time
        self.share_same_time = share_same_time

        kwargs.pop("dataset", None)
        kwargs.pop("collate_fn", None)
        if share_same_time:
            kwargs.pop("sampler", None)
            kwargs["batch_sampler"] = TimestampSampler(
                src_time,
                kwargs["batch_size"],
            )
            kwargs.pop("batch_size", None)

        dataset = CustomLinkDataset(
            self.src_nodes[1],
            dst_nodes[1],
            num_dst_nodes,
            src_time,
        )

        self.src_node_type = self.src_nodes[0]
        self.dst_node_type = self.dst_nodes[0]
        self.src_loader = CustomNodeLoader(
            data,
            node_sampler,
            src_nodes[0],
        )
        self.dst_loader = CustomNodeLoader(
            data,
            node_sampler,
            dst_nodes[0],
        )

        super().__init__(dataset, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(
        self,
        index: Tensor,
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        index = torch.stack(index)
        src_indices = index[:, 0].contiguous()
        pos_dst_indices = index[:, 1].contiguous()
        time = index[:, 2].contiguous()
        neg_dst_indices = torch.randint(0, self.num_dst_nodes, size=(len(src_indices),))
        src_out = self.src_loader.get_neighbors(
            NodeSamplerInput(
                input_id=src_indices,
                node=src_indices,
                time=time,
                input_type=self.src_node_type,
            )
        )

        pos_dst_out = self.dst_loader.get_neighbors(
            NodeSamplerInput(
                input_id=pos_dst_indices,
                node=pos_dst_indices,
                time=time,
                input_type=self.dst_node_type,
            )
        )

        neg_dst_out = self.dst_loader.get_neighbors(
            NodeSamplerInput(
                input_id=neg_dst_indices,
                node=neg_dst_indices,
                time=time,
                input_type=self.dst_node_type,
            )
        )

        return src_out, pos_dst_out, neg_dst_out

