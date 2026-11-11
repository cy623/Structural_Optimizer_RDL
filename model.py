import math
import datetime
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, MLP, SAGEConv as PyGSAGEConv
from torch_scatter import segment_csr
from encoders import HeteroEncoder, HeteroTemporalEncoder



class DeterministicSAGEConv(PyGSAGEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.aggr != 'sum':
            raise ValueError('DeterministicSAGEConv implements the configured sum aggregation only')

    def forward(self, x, edge_index, size=None):
        if not isinstance(edge_index, Tensor):
            raise TypeError('This implementation expects the experiment\'s COO edge_index tensor')
        order = torch.argsort(edge_index[1], stable=True)
        sorted_edges = edge_index[:, order]
        return super().forward(x, sorted_edges, size)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        assert inputs.dim() == 2
        if ptr is None:
            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() else 0
            counts = torch.bincount(index, minlength=dim_size)
            ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        return segment_csr(inputs, ptr, reduce='sum')



SAGEConv = DeterministicSAGEConv


@dataclass(frozen=True)
class Template:
    kind: str
    src: str
    dst: str
    route: tuple = ()


def template_catalog(node_types, edge_types, temporal_fields, strategies):
    if strategies == 'none': return []
    enabled = set(strategies.split(','))
    if enabled == {'all'}: enabled = {'knn','two_hop','behavioral','temporal'}
    if not enabled <= {'knn','two_hop','behavioral','temporal'}:
        raise ValueError('Unknown strategy; use knn,two_hop,behavioral,temporal or all')
    templates = []
    if 'knn' in enabled:
        templates += [Template('knn',nt,nt) for nt in node_types]
    if 'two_hop' in enabled:
        templates += [Template('two_hop',a[0],b[2],(a,b)) for a in edge_types for b in edge_types if a[2]==b[0]]
    if 'behavioral' in enabled:
        templates += [Template('behavioral',e[0],e[0],(e,)) for e in edge_types]
    if 'temporal' in enabled:
        templates += [Template('temporal',nt,nt,(field,)) for nt in node_types for field in temporal_fields.get(nt,[])]
    return templates


def empty(device):
    return torch.empty((2,0),dtype=torch.long,device=device)


@torch.no_grad()
def similarity_edges(x, groups, k=None, threshold=None, eligible=None, budget=1<<24):

    x=F.normalize(x.detach(),dim=-1);device=x.device
    allowed=torch.ones(len(groups),dtype=torch.bool,device=device) if eligible is None else eligible.to(device)
    order=allowed.nonzero().flatten();order=order[torch.sort(groups[order],stable=True).indices]
    _,sizes=torch.unique_consecutive(groups[order],return_counts=True);starts=sizes.cumsum(0)-sizes
    keep=(sizes>=2).nonzero().flatten()
    if not keep.numel():return empty(device)
    keep=keep[torch.sort(sizes[keep],descending=True,stable=True).indices]  # similar sizes share a block
    sizes,starts=sizes[keep],starts[keep];sizes_list=sizes.tolist()
    keys=[];edges=[];i=0
    while i<len(keep):
        n=sizes_list[i];count=max(1,budget//(n*n));sz,st=sizes[i:i+count],starts[i:i+count];i+=count
        slot=torch.arange(n,device=device);valid=slot[None,:]<sz[:,None]
        pos=torch.where(valid,st[:,None]+slot[None,:],0);node=order[pos];feats=x[node]*valid[...,None]
        rows=max(256,budget//(n*len(sz)))
        for r0 in range(0,n,rows):
            r1=min(r0+rows,n);scores=feats[:,r0:r1]@feats.transpose(1,2)
            scores.masked_fill_(~valid[:,None,:],-torch.inf)
            scores[:,torch.arange(r1-r0,device=device),torch.arange(r0,r1,device=device)]=-torch.inf
            query_ok=valid[:,r0:r1]
            if k is not None:
                width=min(k,n-1);top=scores.topk(width,dim=2).indices;rank=torch.arange(width,device=device)
                ok=query_ok[...,None]&(rank[None,None,:]<(sz-1).clamp(max=k)[:,None,None])
                g,q,r=ok.nonzero(as_tuple=True);src=node[g,top[g,q,r]]
                keys.append(pos[g,r0+q]*k+r)
            else:
                g,q,c=((scores>threshold)&query_ok[...,None]).nonzero(as_tuple=True);src=node[g,c]
                keys.append(pos[g,r0+q]*len(order)+pos[g,c])
            edges.append(torch.stack([src,node[g,r0+q]]))
    edges=torch.cat(edges,dim=1)
    return edges[:,torch.argsort(torch.cat(keys))] if edges.numel() else empty(device)


def compose_edges(first,second,shape,budget=1<<25):
    """Exact Boolean two-hop join; no truncation or random edge cap. Unique edges sorted by (row, col)."""
    a,b,c=shape;device=first.device
    if not first.shape[1] or not second.shape[1]:return empty(device)
    mid,perm=torch.sort(second[0],stable=True);far=second[1][perm]
    lo=torch.searchsorted(mid,first[1]);count=torch.searchsorted(mid,first[1],right=True)-lo
    total=count.cumsum(0);last=int(total[-1])  # join in blocks of about `budget` pairs before de-duplication
    cuts=[0]+(torch.searchsorted(total,torch.arange(budget,last,budget,device=device),right=True).tolist()
              if last>budget else [])+[len(count)]
    keys=[]
    for s,e in zip(cuts[:-1],cuts[1:]):
        if s>=e:continue
        n=count[s:e];rep=torch.repeat_interleave(n)
        within=torch.arange(len(rep),device=device)-torch.repeat_interleave(n.cumsum(0)-n,n)
        keys.append(torch.unique(first[0][s:e][rep]*c+far[lo[s:e][rep]+within]))
    key=torch.unique(torch.cat(keys))
    return torch.stack([key//c,key%c])


def temporal_edges(groups,entities,times,m):
    device=groups.device;order=(entities>=0).nonzero().flatten()
    for key in (times,entities,groups):  # lexsort: seed subgraph, FK entity, time, row
        order=order[torch.sort(key[order],stable=True).indices]
    if not order.numel():return empty(device)
    g,e,t=groups[order],entities[order],times[order];pos=torch.arange(len(order),device=device)
    segment=torch.ones_like(pos,dtype=torch.bool);segment[1:]=(g[1:]!=g[:-1])|(e[1:]!=e[:-1])
    run=segment.clone();run[1:]|=t[1:]!=t[:-1]
    segment_start=pos[segment][segment.cumsum(0)-1];run_start=pos[run][run.cumsum(0)-1]
    lo=torch.maximum(segment_start,run_start-m);count=run_start-lo  # rows strictly earlier in time
    target=torch.repeat_interleave(pos,count)
    source=lo[target]+torch.arange(len(target),device=device)-torch.repeat_interleave(count.cumsum(0)-count,count)
    return torch.stack([order[source],order[target]])


@torch.no_grad()
def build_templates(batch,features,templates,cfg,active=None):
    base=batch.edge_index_dict
    result={}
    for i,t in enumerate(templates):
        if active is not None and not active[i]:continue
        if t.kind=='knn':
            edges=similarity_edges(features[t.src],batch[t.src].batch,k=cfg.knn_k,
                                   eligible=batch[t.src].available)
        elif t.kind=='behavioral':
            relation=t.route[0];ei=base.get(relation)
            if ei is None or not ei.numel():continue
            aggregate=torch.zeros_like(features[t.src]);counts=aggregate.new_zeros((len(aggregate),1))
            aggregate.index_add_(0,ei[0],features[relation[2]][ei[1]])
            counts.index_add_(0,ei[0],counts.new_ones((ei.size(1),1)))
            aggregate=aggregate/counts.clamp_min(1)
            edges=similarity_edges(aggregate,batch[t.src].batch,threshold=cfg.sim_threshold,
                                   eligible=(counts[:,0]>0)&batch[t.src].available)
        elif t.kind=='two_hop':
            first,second=t.route
            if first not in base or second not in base:continue
            edges=compose_edges(base[first],base[second],
                (len(features[first[0]]),len(features[first[2]]),len(features[second[2]])))
        else:
            edges=temporal_edges(batch[t.src].batch,batch[t.src][t.route[0]],batch[t.src].time,cfg.recent_k)
        if edges.numel():result[i]=edges
    return result


class HardConcrete(nn.Module):
    def __init__(self,n,init=2.,tau=2.):
        super().__init__();self.logit=nn.Parameter(torch.full((n,),float(init)))
        self.register_buffer('tau',torch.tensor(float(tau)))
    def forward(self,threshold=.5):
        if not self.training:return (self.logit.sigmoid()>threshold).to(self.logit.dtype)
        u=torch.rand_like(self.logit).clamp(1e-6,1-1e-6)
        return ((torch.sigmoid((u.log()-(-u).log1p()+self.logit)/self.tau)*1.2)-.1).clamp(0,1)
    def l0(self):return torch.sigmoid(self.logit-self.tau*math.log(.1/1.1)).sum()


class TypeVIB(nn.Module):
    def __init__(self,channels):
        super().__init__();self.mu=nn.Linear(channels,channels);self.logvar=nn.Linear(channels,channels)
        nn.init.zeros_(self.logvar.weight);nn.init.constant_(self.logvar.bias,-2.)
    def forward(self,x):
        if not x.numel():return x,x.new_zeros(())
        mu=self.mu(x);logv=self.logvar(x).clamp(-10,10)
        z=mu+torch.randn_like(mu)*(logv*.5).exp() if self.training else mu
        kl=.5*(logv.exp()+mu.square()-1-logv).sum(-1).mean()
        return z,kl


class StructuralOptimizer(nn.Module):
    def __init__(self,data,stats,cfg,temporal_fields):
        super().__init__();self.cfg=cfg;self.types=list(data.node_types);self.relations=list(data.edge_types)
        self.encoder=HeteroEncoder(cfg.channels,{nt:data[nt].tf.col_names_dict for nt in self.types},stats)
        self.temporal_encoder=HeteroTemporalEncoder([nt for nt in self.types if 'time' in data[nt]],cfg.channels)
        self.column_gates=nn.ModuleDict({nt:HardConcrete(cfg.channels,cfg.hc_init_logit,cfg.hc_tau_start) for nt in self.types})
        self.table_gates=nn.ModuleDict({nt:HardConcrete(1,cfg.hc_init_logit,cfg.hc_tau_start) for nt in self.types})
        self.vib=nn.ModuleDict({nt:TypeVIB(cfg.channels) for nt in self.types}) if cfg.beta_vib>0 else None
        self.templates=template_catalog(self.types,self.relations,temporal_fields,cfg.strategies)
        self.structure_gate=HardConcrete(len(self.templates),cfg.hc_init_logit,cfg.hc_tau_start)
        self.base_convs=nn.ModuleList([nn.ModuleList([SAGEConv((cfg.channels,cfg.channels),cfg.channels,aggr=cfg.aggr)
            for _ in self.relations]) for _ in range(cfg.num_layers)])
        self.template_convs=nn.ModuleList([nn.ModuleList([SAGEConv((cfg.channels,cfg.channels),cfg.channels,
            aggr=cfg.aggr,root_weight=False,bias=False) for _ in self.templates]) for _ in range(cfg.num_layers)])
        self.norms=nn.ModuleList([nn.ModuleDict({nt:LayerNorm(cfg.channels,mode='node') for nt in self.types}) for _ in range(cfg.num_layers)])
        self.head=MLP(cfg.channels,out_channels=1,num_layers=1,norm='batch_norm')
        self.last_edges={}

    def anneal(self,epoch):
        cfg=self.cfg;fraction=max(0.,(epoch-cfg.warmup_epochs)/max(1,cfg.epochs-cfg.warmup_epochs))
        tau=cfg.hc_tau_start*(cfg.hc_tau_end/cfg.hc_tau_start)**fraction
        for module in self.modules():
            if isinstance(module,HardConcrete):module.tau.fill_(tau)

    def forward(self,batch,entity):
        cfg=self.cfg;seed_time=batch[entity].seed_time;initial={}
        for nt in self.types:
            store=batch[nt];mask=store.available
            h0=self.head.lins[0].weight.new_zeros((len(mask),cfg.channels))
            if bool(mask.any()):
                h0[mask]=self.encoder.encoders[nt](store.tf[mask])
            initial[nt]=h0
        valid_times={nt:batch[nt].time[batch[nt].available] for nt in batch.time_dict}
        valid_groups={nt:batch[nt].batch[batch[nt].available] for nt in valid_times}
        for nt,relative in self.temporal_encoder(seed_time,valid_times,valid_groups).items():
            initial[nt][batch[nt].available]=initial[nt][batch[nt].available]+relative
        reg={key:seed_time.new_zeros((),dtype=torch.float) for key in ['l0_w','l0_g','l0_h','kl_vib']}
        x={}
        for nt,h0 in initial.items():
            column=self.column_gates[nt](cfg.gate_threshold) if cfg.lambda_w>0 else h0.new_ones(cfg.channels)
            table=self.table_gates[nt](cfg.gate_threshold) if cfg.lambda_g>0 else h0.new_ones(1)
            filtered=h0*column*table
            reg['l0_w']=reg['l0_w']+self.column_gates[nt].l0()
            reg['l0_g']=reg['l0_g']+self.table_gates[nt].l0()
            if self.vib is not None:
                mask=batch[nt].available
                valid,kl=self.vib[nt](filtered[mask])
                x[nt]=torch.zeros_like(filtered)
                x[nt][mask]=valid
                reg['kl_vib']=reg['kl_vib']+kl
            else:x[nt]=filtered
        h=self.structure_gate(cfg.gate_threshold);reg['l0_h']=self.structure_gate.l0()
        edges=build_templates(batch,initial,self.templates,cfg,None if self.training else h.detach().bool().cpu().tolist())
        self.last_edges=edges
        for layer in range(cfg.num_layers):
            out={nt:torch.zeros_like(value) for nt,value in x.items()}
            for relation,conv in zip(self.relations,self.base_convs[layer]):
                if relation in batch.edge_index_dict:
                    src,_,dst=relation;out[dst]=out[dst]+conv((x[src],x[dst]),batch[relation].edge_index)
            for idx,ei in edges.items():
                template=self.templates[idx]
                out[template.dst]=out[template.dst]+h[idx]*self.template_convs[layer][idx]((x[template.src],x[template.dst]),ei)
            x={nt:self.norms[layer][nt](value) for nt,value in out.items()}
            if layer+1<cfg.num_layers:x={nt:value.relu() for nt,value in x.items()}
        logits=self.head(x[entity][:seed_time.numel()]).flatten()
        return logits,reg

    def regularization(self,reg):
        c=self.cfg
        return c.lambda_w*reg['l0_w']+c.lambda_g*reg['l0_g']+c.lambda_h*reg['l0_h']+c.beta_vib*reg['kl_vib']

    def template_manifest(self):return [asdict(t) for t in self.templates]

    @torch.no_grad()
    def gate_signature(self,threshold):
        assert not self.training
        return b''.join(m(threshold).cpu().numpy().tobytes() for m in self.modules() if isinstance(m,HardConcrete))


class Model(StructuralOptimizer):

    def __init__(self, args, data, col_stats_dict, num_layers=None, channels=None,
                 out_channels=1, aggr=None, norm="batch_norm", shallow_list=None,
                 id_awareness=False, *, temporal_fields=None, config=None):
        from types import SimpleNamespace
        if out_channels != 1 or norm != "batch_norm" or shallow_list or id_awareness:
            raise ValueError("This version implements the validated binary-classification model without ID embeddings.")
        values = dict(config) if config is not None else dict(vars(args))
        if config is None:
            aliases = {"2hop_shot": "two_hop", "2hop_agg": "behavioral", "recent_k": "temporal"}
            strategies = values.get("struct_strategies", values.get("strategies", "all"))
            values["strategies"] = ",".join(dict.fromkeys(aliases.get(item.strip(), item.strip()) for item in strategies.split(",")))
            values.setdefault("gate_threshold", 0.5)
            values.setdefault("sim_threshold", 0.5)
        for name, value in [("num_layers", num_layers), ("channels", channels), ("aggr", aggr)]:
            if value is not None:
                values[name] = value
        if temporal_fields is None:
            temporal_fields = {}
            for node_type in data.node_types:
                fields = [key for key in data[node_type].keys() if key.startswith("temporal_entity_")]
                if fields:
                    temporal_fields[node_type] = sorted(fields, key=lambda key: int(key.rsplit("_", 1)[1]))
            if values["strategies"] in {"all", "temporal"} or "temporal" in values["strategies"].split(","):
                if any("time" in data[nt] for nt in data.node_types) and not temporal_fields:
                    raise ValueError("Temporal FK identity is missing; prepare the graph through the supplied main.py.")
        super().__init__(data, col_stats_dict, SimpleNamespace(**values), temporal_fields)
        self.temporal_fields = temporal_fields
        self.args = args

    def forward(self, batch, entity_table, apply_pruning=None):
        return super().forward(batch, entity_table)

    def update_anneal(self, epoch, epochs, warmup_epochs, tau_start, tau_end):
        self.cfg.epochs = epochs
        self.cfg.warmup_epochs = warmup_epochs
        self.cfg.hc_tau_start = tau_start
        self.cfg.hc_tau_end = tau_end
        self.anneal(epoch)


