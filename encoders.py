from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch
import torch_frame
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP, HeteroConv, LayerNorm, PositionalEncoding, SAGEConv
from torch_geometric.typing import NodeType, EdgeType
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_scatter import scatter
import torch.nn.functional as F


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
    


class HeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype, List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype: default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1]
                )
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf) for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict(
            {node_type: PositionalEncoding(channels) for node_type in node_types}
        )
        self.lin_dict = torch.nn.ModuleDict(
            {node_type: torch.nn.Linear(channels, channels) for node_type in node_types}
        )

    def reset_parameters(self):
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        num_sampled_nodes_dict: Optional[Dict[str, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[Tuple[str, str, str], List[int]]] = None,
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        
        # 关键：保持向后兼容 - 没有边权重时使用原始逻辑
        if edge_weight_dict is None or len(edge_weight_dict) == 0:
            return self._original_forward(x_dict, edge_index_dict)
        
        # 有边权重时使用新逻辑
        return self._forward_with_weights(x_dict, edge_index_dict, edge_weight_dict)
    
    def _original_forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict
    
    def _forward_with_weights(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_weight_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = self._message_passing_with_weights(
                conv, x_dict, edge_index_dict, edge_weight_dict, i
            )
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

    def _message_passing_with_weights(self, conv, x_dict, edge_index_dict, edge_weight_dict, layer_idx):
        out_dict = {node_type: torch.zeros_like(x_dict[node_type]) for node_type in self.node_types}
        
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
                
            src_type, rel_type, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]
            
            if edge_index.size(1) == 0:
                continue
            
            edge_weight = edge_weight_dict.get(edge_type)
            sage_conv = conv.convs[edge_type]
            
            if edge_weight is not None:
                x_src = x_dict[src_type]
                x_dst = x_dict[dst_type]
                
                if sage_conv.project:
                    x_src = sage_conv.lin_src(x_src)
                    if x_dst is not None:
                        x_dst = sage_conv.lin_dst(x_dst)
                
                messages = x_src[edge_index[0]]
                if edge_weight is not None:
                    messages = messages * edge_weight.unsqueeze(1)
                
                aggregated = scatter(messages, edge_index[1], dim=0, 
                                   dim_size=x_dst.size(0), reduce=sage_conv.aggr)
                
                if sage_conv.root_weight and x_dst is not None:
                    aggregated = aggregated + x_dst
                
                out_dict[dst_type] = out_dict[dst_type] + aggregated
            else:
                x_dst = conv({src_type: x_dict[src_type]}, edge_index, node_type=dst_type)
                out_dict[dst_type] = out_dict[dst_type] + x_dst[dst_type]
        
        return out_dict



# class HeteroGraphSAGE(torch.nn.Module):
#     def __init__(
#         self,
#         node_types: List[NodeType],
#         edge_types: List[EdgeType],
#         channels: int,
#         aggr: str = "mean",
#         num_layers: int = 2,
#     ):
#         super().__init__()

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             conv = HeteroConv(
#                 {
#                     edge_type: SAGEConv((channels, channels), channels, aggr=aggr)
#                     for edge_type in edge_types
#                 },
#                 aggr="sum",
#             )
#             self.convs.append(conv)

#         self.norms = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             norm_dict = torch.nn.ModuleDict()
#             for node_type in node_types:
#                 norm_dict[node_type] = LayerNorm(channels, mode="node")
#             self.norms.append(norm_dict)

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for norm_dict in self.norms:
#             for norm in norm_dict.values():
#                 norm.reset_parameters()

#     def forward(
#         self,
#         x_dict: Dict[NodeType, Tensor],
#         edge_index_dict: Dict[NodeType, Tensor],
#         num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
#         num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
#     ) -> Dict[NodeType, Tensor]:
#         for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
#             x_dict = conv(x_dict, edge_index_dict)
#             x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
#             x_dict = {key: x.relu() for key, x in x_dict.items()}

#         return x_dict
    


