from typing import List, Optional, Dict, Any, Tuple
from torch import Tensor
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from torch_frame.data.stats import StatType
from encoders import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from torch.nn import Embedding, ModuleDict
from relgnn import RelGNN
import torch.nn as nn
from gates import StructureGate
from collections.abc import Mapping
import math
from copy import deepcopy


class L0Gate(nn.Module):
    def __init__(self, n, init_logit=2.0, tau=2.0, gamma=-0.1, zeta=1.1, eps=1e-6):
        super().__init__()
        self.logit = nn.Parameter(torch.full((n,), float(init_logit)))
        self.tau = float(tau)
        self.gamma, self.zeta = float(gamma), float(zeta)
        self.eps = float(eps)
        self.n = n 

    def set_tau(self, tau: float):
        self.tau = float(max(tau, 1e-3))  

    def forward(self):
        if self.training:
            u = torch.rand_like(self.logit)
            u = u.clamp(self.eps, 1.0 - self.eps)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logit) / self.tau)
        else:
            s = torch.sigmoid(self.logit / self.tau)

        y = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(y, 0.0, 1.0)
    
    def get_discrete_mask(self, threshold=0.5):
        with torch.no_grad():
            s = torch.sigmoid(self.logit / self.tau)
            return (s > threshold).float()
    
    def get_active_count(self, threshold=0.5):
        mask = self.get_discrete_mask(threshold)
        return mask.sum().item()

    def l0(self):
        c = math.log((0.0 - self.gamma) / (self.zeta - 0.0) + self.eps)
        t = self.logit - self.tau * c
        return torch.sigmoid(t).sum()
    

class TypeVIB(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.mu = nn.Linear(in_dim, hid)
        self.logvar = nn.Linear(in_dim, hid)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight); nn.init.constant_(self.logvar.bias, -2.0) 

    def forward(self, x: torch.Tensor):
        if x.numel() == 0:
            z = x  
            kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return z, kl

        amp_enabled = torch.is_autocast_enabled()
        with torch.autocast(device_type=x.device.type if x.is_cuda else 'cpu', enabled=False):
            x32 = x.float()
            mu = self.mu(x32)
            logv = self.logvar(x32)

            logv = torch.clamp(logv, min=-10.0, max=10.0)  
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            z32 = mu + eps * std

            kl = 0.5 * torch.mean(torch.sum(torch.exp(logv) + mu * mu - 1.0 - logv, dim=-1))

        z = z32.to(dtype=x.dtype)
        kl = kl.to(dtype=x.dtype)
        return z, kl


def build_knn_edges(x: Tensor, k: int) -> Tensor:
    if x.size(0) == 0 or k <= 0:
        return None
    x = nn.functional.normalize(x, p=2, dim=-1)
    sim = x @ x.t()                           # [N, N]
    N = x.size(0)
    topk = torch.topk(sim, k=min(k + 1, N), dim=-1).indices  # [N, k+1]
    src = torch.arange(N, device=x.device).unsqueeze(-1).expand_as(topk)  # [N, k+1]
    src = src.reshape(-1)
    dst = topk.reshape(-1)
    mask = (src != dst)
    ei = torch.stack([src[mask], dst[mask]], dim=0)
    return ei

def two_hop_shortcuts(e12: Tensor, e23: Tensor) -> Tensor:
    if e12 is None or e23 is None or e12.numel() == 0 or e23.numel() == 0:
        return None
    a_of_b = {}
    src, mid = e12[0].tolist(), e12[1].tolist()
    for a, b in zip(src, mid):
        a_of_b.setdefault(b, []).append(a)
    mids, cands = e23[0].tolist(), e23[1].tolist()
    A, C = [], []
    for b, c in zip(mids, cands):
        if b in a_of_b:
            for a in a_of_b[b]:
                A.append(a); C.append(c)
    if len(A) == 0:
        return None
    return torch.tensor([A, C], dtype=torch.long, device=e12.device)





class Model(torch.nn.Module):

    def __init__(
        self,
        args,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.args = args
        self.data = data
        self.channels = channels
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.threshold = 0.5 
        
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
       
        self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.enable_type_gate = bool(args.lambda_g > 0)
        self.enable_col_gate  = bool(args.lambda_g > 0)
        self.enable_struct    = bool(args.lambda_h > 0)
        self.enable_vib       = bool(args.beta_vib > 0)

        if self.enable_type_gate:
            self.type_gate = ModuleDict({
                nt: L0Gate(1, init_logit=args.hc_init_logit, tau=args.hc_tau_start) for nt in self.node_types
            })
        else:
            self.type_gate = None

        if self.enable_col_gate:
            self.col_gate = ModuleDict()
            for nt in self.node_types:
                d = 0
                for _, cols in data[nt].tf.col_names_dict.items():
                    d += len(cols)
                self.col_gate[nt] = L0Gate(d, init_logit=args.hc_init_logit, tau=args.hc_tau_start)
        else:
            self.col_gate = None

        if self.enable_vib:
            self.vib = ModuleDict({nt: TypeVIB(in_dim=channels, hid=channels) for nt in self.node_types})
        else:
            self.vib = None

        if self.enable_struct:
            self.struct_templates: List[str] = []
            for nt in self.node_types:
                self.struct_templates.append(f"knn::{nt}")
            for (A, r1, B) in self.edge_types:
                for (B2, r2, C) in self.edge_types:
                    if B2 == B:
                        self.struct_templates.append(f"2hop::{A}->{B}->{C}")
            self.h_gate = L0Gate(len(self.struct_templates), init_logit=args.hc_init_logit, tau=args.hc_tau_start)
        else:
            self.struct_templates = []
            self.h_gate = None

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
    
    def get_pruning_stats(self, threshold=0.5):
        stats = {
            "table_decisions": {},
            "col_decisions": {},
            "struct_decisions": None
        }
        
        if self.enable_type_gate:
            for nt, gate in self.type_gate.items():
                mask = gate.get_discrete_mask(threshold)
                stats["table_decisions"][nt] = mask.item() > 0.5
        
        if self.enable_col_gate:
            for nt, gate in self.col_gate.items():
                mask = gate.get_discrete_mask(threshold)
                active = mask.sum().item()
                total = mask.numel()
                stats["col_decisions"][nt] = {
                    "active": active,
                    "total": total,
                    "ratio": active / total if total > 0 else 0.0
                }
        
        if self.enable_struct:
            mask = self.h_gate.get_discrete_mask(threshold)
            stats["struct_decisions"] = {
                "active": mask.sum().item(),
                "total": mask.numel()
            }
        
        return stats
    
    def create_pruned_batch(self, original_batch: HeteroData, threshold=0.5) -> HeteroData:
 
        
        pruned_batch = deepcopy(original_batch)

        stats = self.get_pruning_stats(threshold)

        if self.enable_type_gate:
            for node_type, keep in stats["table_decisions"].items():
                if not keep:
                    
                    if node_type in pruned_batch.node_types:
                        
                        if hasattr(pruned_batch[node_type], 'x'):
                            delattr(pruned_batch[node_type], 'x')
                        
                        pruned_batch.node_types = [nt for nt in pruned_batch.node_types if nt != node_type]
                                   
                        edges_to_remove = []
                        for edge_type in list(pruned_batch.edge_types):
                            src, rel, dst = edge_type
                            if src == node_type or dst == node_type:
                                edges_to_remove.append(edge_type)
                        
                        for edge_type in edges_to_remove:
                            del pruned_batch[edge_type]
        
        if self.enable_col_gate:
            for node_type, col_info in stats["col_decisions"].items():
                if node_type not in pruned_batch.node_types:
                    continue
                    
                if hasattr(pruned_batch[node_type], 'x'):
                    original_x = pruned_batch[node_type].x
                    if col_info["active"] > 0:
                        
                        pass

        if self.enable_struct and stats["struct_decisions"]:
            h_mask = self.h_gate.get_discrete_mask(threshold)
            activated_templates = []
            for idx, name in enumerate(self.struct_templates):
                if h_mask[idx].item() > 0.5:
                    activated_templates.append(name)

            for node_type in self.node_types:
                if node_type not in pruned_batch.node_types:
                    continue
                    
                knn_key = (node_type, f"knn_{node_type}", node_type)
                if knn_key in pruned_batch.edge_index_dict:
                    if f"knn::{node_type}" not in activated_templates:
                        del pruned_batch[knn_key]
            
            for (A, r1, B) in self.edge_types:
                for (B2, r2, C) in self.edge_types:
                    if B2 == B:
                        key = (A, f"2hop_{A}_{B}_{C}", C)
                        template_name = f"2hop::{A}->{B}->{C}"
                        if key in pruned_batch.edge_index_dict and template_name not in activated_templates:
                            del pruned_batch[key]
        
        return pruned_batch
    

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        apply_pruning: bool = False,
    ) -> Tensor:
       
        seed_time = batch[entity_table].seed_time
        device = seed_time.device

        if apply_pruning and not self.training:
    
            batch = self.create_pruned_batch(batch, self.threshold)

            x_dict = self.encoder(batch.tf_dict)

            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

            for node_type, embedding in self.embedding_dict.items():
                x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

            edge_index_dict = dict(batch.edge_index_dict)  

            if self.enable_struct and len(self.struct_templates) > 0:
                h_mask = self.h_gate.get_discrete_mask(self.threshold)  # [num_templates]

                for idx, name in enumerate(self.struct_templates):
                    if h_mask[idx].item() <= 0.5:
                        continue

                    if name.startswith("knn::"):
                        nt = name.split("::", 1)[1]
                        if nt in x_dict and nt in batch.node_types:
                            ei = build_knn_edges(x_dict[nt], k=self.args.knn_k)
                            if ei is not None:
                                key = (nt, "knn_" + nt, nt)
                                edge_index_dict[key] = ei.to(device)

                    elif name.startswith("2hop::"):
                        trip = name[len("2hop::"):]
                        A, rest = trip.split("->", 1)
                        B, C = rest.split("->", 1)

                        e12, e23 = None, None
                        for (s, r, t), ei in batch.edge_index_dict.items():
                            if s == A and t == B:
                                e12 = ei
                            if s == B and t == C:
                                e23 = ei
                        sc = two_hop_shortcuts(e12, e23)
                        if sc is not None:
                            key = (A, f"2hop_{A}_{B}_{C}", C)
                            edge_index_dict[key] = sc.to(device)

            x_dict = self.gnn(
                x_dict,
                edge_index_dict,
                batch.num_sampled_nodes_dict,
                batch.num_sampled_edges_dict,
            )
            logits = self.head(x_dict[entity_table][: seed_time.size(0)])

            zero = torch.tensor(0.0, device=device)
            reg_terms = {
                "kl_vib": zero,
                "l0_g":   zero,
                "l0_w":   zero,
                "l0_h":   zero,
                "h_activated": zero,
            }
            return logits, reg_terms

        x_dict = self.encoder(batch.tf_dict)  # get the original features based on raw datas

        # get time feature based on timestamp
        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        reg_terms = {
            "kl_vib": torch.tensor(0.0, device=seed_time.device),
            "l0_g":   torch.tensor(0.0, device=seed_time.device),
            "l0_w":   torch.tensor(0.0, device=seed_time.device),
            "l0_h":   torch.tensor(0.0, device=seed_time.device),
            "h_activated": torch.tensor(0.0, device=seed_time.device),
        }

        if self.enable_col_gate:
            for nt, x in x_dict.items():
                g = self.col_gate[nt]()  # [num_cols] in [0,1]
                repeat = math.ceil(self.channels / g.numel())
                mask = g.repeat_interleave(1).repeat(repeat)[: self.channels]  # [C]
                x_dict[nt] = x * mask
                reg_terms["l0_w"] = reg_terms["l0_w"] + self.col_gate[nt].l0()

        if self.enable_type_gate:
            for nt, x in x_dict.items():
                g = self.type_gate[nt]() 
                x_dict[nt] = x * g
                reg_terms["l0_g"] = reg_terms["l0_g"] + self.type_gate[nt].l0()


        if self.enable_vib:
            kl_sum = torch.tensor(0.0, device=seed_time.device, dtype=x_dict[next(iter(x_dict))].dtype)
            for nt, x in x_dict.items():
                if x.numel() == 0:
                    continue
                z, kl = self.vib[nt](x)
                x_dict[nt] = z
                kl_sum = kl_sum + kl
            reg_terms["kl_vib"] = kl_sum


        edge_index_dict = dict(batch.edge_index_dict)  

        if self.enable_struct and len(self.struct_templates) > 0:
            h = self.h_gate()   # [num_templates] \in [0,1]
            reg_terms["l0_h"] = self.h_gate.l0()
            reg_terms["h_activated"] = h.sum()

            for idx, name in enumerate(self.struct_templates):
                if float(h[idx].item()) <= 0.0:
                    continue
                if name.startswith("knn::"):
                    nt = name.split("::", 1)[1]
                    if nt in x_dict:
                        ei = build_knn_edges(x_dict[nt], k=self.args.knn_k)
                        if ei is not None:
                            key = (nt, "knn_" + nt, nt)
                            edge_index_dict[key] = ei.to(x_dict[nt].device)
                elif name.startswith("2hop::"):
                    trip = name[len("2hop::"):]
                    A, rest = trip.split("->", 1)
                    B, C = rest.split("->", 1)
                    e12, e23 = None, None
                    for (s, r, t), ei in batch.edge_index_dict.items():
                        if s == A and t == B:
                            e12 = ei
                        if s == B and t == C:
                            e23 = ei
                    sc = two_hop_shortcuts(e12, e23)
                    if sc is not None:
                        key = (A, f"2hop_{A}_{B}_{C}", C)
                        edge_index_dict[key] = sc.to(seed_time.device)


        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        logits = self.head(x_dict[entity_table][: seed_time.size(0)])

        return logits, reg_terms

    

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
    


class RelGNN_Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_model_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
        atomic_routes=None,
        num_heads=None,
        simplified_MP=False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = RelGNN(
            node_types=data.node_types,
            edge_types=atomic_routes,
            channels=channels,
            aggr=aggr,
            num_model_layers=num_model_layers,
            num_heads=num_heads,
            simplified_MP=simplified_MP,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])




class StructuralInjectionManager(nn.Module):
    
    def __init__(self, channels, node_types, base_edge_types, args):
        super().__init__()
        self.channels = channels
        self.node_types = node_types
        self.base_edge_types = base_edge_types
        self.args = args
        
        self.available_strategies = {
            'knn': self._build_knn_templates,
            'two_hop': self._build_two_hop_templates, 
            'behavioral': self._build_behavioral_templates,
            'temporal': self._build_temporal_templates
        }
        
        self.active_strategies = self._parse_strategy_config(args.struct_strategies)
        
        self._init_templates_and_gates()
    
    def _parse_strategy_config(self, strategy_config):
        if strategy_config is None or strategy_config == "none":
            return []
        elif strategy_config == "all":
            return list(self.available_strategies.keys())
        elif isinstance(strategy_config, str):
            return [s.strip() for s in strategy_config.split(',')]
        elif isinstance(strategy_config, list):
            return strategy_config
        else:
            return []
    
    def _init_templates_and_gates(self):
        self.templates = []
        self.aug_edge_types = []
        
        for strategy in self.active_strategies:
            if strategy in self.available_strategies:
                template_builder = self.available_strategies[strategy]
                template_builder()
        
        if self.templates:
            self.template_gates = L0Gate(
                len(self.templates),
                init_logit=self.args.hc_init_logit,
                tau=self.args.hc_tau_start
            )
        else:
            self.template_gates = None
    
    def _build_knn_templates(self):
        for node_type in self.node_types:
            template_name = f"knn::{node_type}"
            self.templates.append({
                'name': template_name,
                'type': 'knn',
                'node_type': node_type,
                'k': getattr(self.args, 'knn_k', 5)
            })
            self.aug_edge_types.append((node_type, f"knn_{node_type}", node_type))
    
    def _build_two_hop_templates(self):
        for (A, r1, B) in self.base_edge_types:
            for (B2, r2, C) in self.base_edge_types:
                if B2 == B:
                    template_name = f"2hop::{A}->{B}->{C}"
                    self.templates.append({
                        'name': template_name,
                        'type': '2hop',
                        'path': [A, B, C],
                        'relations': [r1, r2]
                    })
                    self.aug_edge_types.append((A, f"2hop_{A}_{B}_{C}", C))
    
    def _build_behavioral_templates(self):
        for src_type in self.node_types:
            outgoing_edges = [et for et in self.base_edge_types if et[0] == src_type]
            if outgoing_edges:
                template_name = f"behavioral::{src_type}"
                self.templates.append({
                    'name': template_name,
                    'type': 'behavioral',
                    'src_type': src_type
                })
                self.aug_edge_types.append((src_type, f"behavioral_{src_type}", src_type))
    
    def _build_temporal_templates(self):
        temporal_nodes = [nt for nt in self.node_types if hasattr(self.args, 'temporal_nodes') and nt in self.args.temporal_nodes]
        for node_type in temporal_nodes:
            template_name = f"temporal::{node_type}"
            self.templates.append({
                'name': template_name,
                'type': 'temporal',
                'node_type': node_type,
                'k': getattr(self.args, 'recent_k', 3)
            })
            self.aug_edge_types.append((node_type, f"temporal_{node_type}", node_type))
    
    def forward(self, x_dict, batch, training=True):
        if not self.templates or self.template_gates is None:
            return {}, {}, 0.0
        
        sample_device = next(iter(x_dict.values())).device
        gate_values = self.template_gates().to(sample_device) 
        selected_edges = {}
        edge_weights = {}
        
        for idx, template in enumerate(self.templates):
            gate_val = gate_values[idx]
            
            threshold = 1e-3 if training else 0.5
            if gate_val <= threshold:
                continue
                
            edges = self._generate_edges_for_template(template, x_dict, batch)
            if edges is not None and edges.size(1) > 0:
                edge_key = self._get_edge_key_for_template(template)
                selected_edges[edge_key] = edges
                
                edge_weights[edge_key] = gate_val * torch.ones(
                    edges.size(1), device=edges.device 
                )
        
        l0_penalty = self.template_gates.l0() if self.template_gates else 0.0
        
        return selected_edges, edge_weights, l0_penalty

    
    def _generate_edges_for_template(self, template, x_dict, batch):
        template_type = template['type']
        
        if template_type == 'knn':
            return self._generate_knn_edges(template, x_dict)
        elif template_type == '2hop':
            return self._generate_two_hop_edges(template, batch)
        elif template_type == 'behavioral':
            return self._generate_behavioral_edges(template, x_dict, batch)
        elif template_type == 'temporal':
            return self._generate_temporal_edges(template, x_dict, batch)
        else:
            return None
    
    def _generate_knn_edges(self, template, x_dict):
        node_type = template['node_type']
        k = template['k']
        
        if node_type not in x_dict:
            return None
            
        features = x_dict[node_type]
        if features.size(0) < 2:
            return None
        
        n_nodes = features.size(0)
        device = features.device
        src = torch.arange(n_nodes, device=device).repeat_interleave(k)
        offsets = torch.arange(1, k+1, device=device).repeat(n_nodes)
        dst = (src + offsets) % n_nodes
        
        return torch.stack([src, dst])
    
    
    def _generate_two_hop_edges(self, template, batch):
        A, B, C = template['path']
        
        e12, e23 = None, None
        for edge_type, edges in batch.edge_index_dict.items():
            if edge_type[0] == A and edge_type[2] == B:
                e12 = edges
            elif edge_type[0] == B and edge_type[2] == C:
                e23 = edges
        
        if e12 is None or e23 is None:
            return None
        
        return two_hop_shortcuts(e12, e23)
    
    def _generate_behavioral_edges(self, template, x_dict, batch):
        src_type = template['src_type']
        
        if src_type not in x_dict:
            return None
            
        features = x_dict[src_type]
        if features.size(0) < 2:
            return None
        
        device = features.device
        n_nodes = features.size(0)
        src = torch.arange(n_nodes, device=device)
        dst = torch.randperm(n_nodes, device=device) 
        
        return torch.stack([src, dst])
    
    def _generate_temporal_edges(self, template, x_dict, batch):
        node_type = template['node_type']
        k = template['k']
        
        if node_type not in x_dict:
            return None
            
        n_nodes = x_dict[node_type].size(0)
        if n_nodes < 2:
            return None
        
        device = x_dict[node_type].device 

        src = torch.arange(n_nodes - 1, device=device)
        dst = torch.arange(1, n_nodes, device=device)
        
        return torch.stack([src, dst])
    
    def _get_edge_key_for_template(self, template):
        template_type = template['type']
        
        if template_type == 'knn':
            node_type = template['node_type']
            return (node_type, f"knn_{node_type}", node_type)
        elif template_type == '2hop':
            A, B, C = template['path']
            return (A, f"2hop_{A}_{B}_{C}", C)
        elif template_type == 'behavioral':
            src_type = template['src_type']
            return (src_type, f"behavioral_{src_type}", src_type)
        elif template_type == 'temporal':
            node_type = template['node_type']
            return (node_type, f"temporal_{node_type}", node_type)
        
        return None
    
    def get_augmented_edge_types(self):
        return self.aug_edge_types