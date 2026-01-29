import torch
import tqdm
from torch.nn import BCEWithLogitsLoss, L1Loss
import numpy as np
from model import Model, RelGNN_Model
import os 
import math
from data.task_base import TaskType
from enum import Enum
from data.utils import get_configs
from atomic_routes import get_atomic_routes
from sklearn.metrics import roc_auc_score, average_precision_score
import time
_SK_AVAILABLE = True



class Experiment:
    def __init__(self, args, data, col_stats_dict, entity_table, loader_dict, task, device):
        self.args = args
        self.data = data
        self.col_stats_dict = col_stats_dict
        self.entity_table = entity_table
        self.loader_dict = loader_dict
        self.task = task
        self.device = device
        self.work_dir = ' '
        os.makedirs(self.work_dir, exist_ok=True)
        self.task_type = self.normalize_task_type(task.task_type, TaskType)

        self.out_channels, self.loss_fn, self.tune_metric, self.higher_is_better = self._init_task_cfg(task)
        self.clamp_min, self.clamp_max = self._compute_clamp_bounds_if_needed(task)
        
        if args.model == 'RelGNN':
            model_config, loader_config = get_configs(args.dataset, args.task)
            atomic_routes_list = get_atomic_routes(self.data.edge_types)
            self.model = RelGNN_Model(
            data=self.data,
            col_stats_dict=self.col_stats_dict,
            out_channels=self.out_channels,
            norm="batch_norm",
            atomic_routes=atomic_routes_list,
            **model_config,
        ).to(device)
            
        if args.model == 'HeteroGraphSAGE':
            self.model = Model(
                args=self.args,
                data=self.data,
                col_stats_dict=self.col_stats_dict,
                num_layers=args.num_layers,
                channels=args.channels,
                out_channels=self.out_channels,
                aggr=args.aggr,
                norm="batch_norm",
            ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        self.epochs = args.epochs
        self.best_metric = -math.inf if self.higher_is_better else math.inf
        self.best_path = os.path.join(self.work_dir, f"{args.task}{args.struct_strategies}best.pt")

        self.enable_type_gate = bool(args.lambda_g > 0)
        self.enable_col_gate  = bool(args.lambda_g > 0)
        self.enable_struct    = bool(args.lambda_h > 0)
        self.enable_vib       = bool(args.beta_vib > 0)

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            if hasattr(self.model, "update_anneal"):
                self.model.update_anneal(epoch, self.epochs, self.args.warmup_epochs,
                                         self.args.hc_tau_start, self.args.hc_tau_end)
                
            train_loss = self._train_one_epoch(epoch)
            val_metric = self._evaluate(split="val")

            improved = (val_metric > self.best_metric) if self.higher_is_better else (val_metric < self.best_metric)
            if improved:
                self.best_metric = val_metric
                torch.save(self.model.state_dict(), self.best_path)

            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_{self.tune_metric}={val_metric:.4f}  "
                  f"{'↑' if self.higher_is_better else '↓'} best={self.best_metric:.4f}")
            

    def test(self):
        if os.path.exists(self.best_path):
            self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        test_metric = self._evaluate(split="test")
        print(f"[Test] {self.tune_metric}={test_metric:.4f}")
        return test_metric
    
            
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        loader = self.loader_dict["train"]
        running_loss = 0.0
        n = 0
        total_steps = min(len(loader), self.args.max_steps_per_epoch)
        pbar = tqdm.tqdm(loader, total=total_steps)
        for batch in pbar:
            batch = batch.to(self.device)

            out = self.model(batch, self.entity_table)
           
            if isinstance(out, tuple):
                logits, reg_terms = out
            else:
                logits, reg_terms = out, None

            target = self._extract_target(batch)
            task_loss = self._compute_loss(logits, target)

            sib_loss = 0.0
            if reg_terms is not None:
                if self.enable_vib:
                    sib_loss += self.args.beta_vib * reg_terms.get("kl_vib", 0.0)
                if self.enable_type_gate:
                    sib_loss += self.args.lambda_g * reg_terms.get("l0_g", 0.0)
                if self.enable_col_gate:
                    sib_loss += self.args.lambda_w * reg_terms.get("l0_w", 0.0)
                if self.enable_struct:
                    sib_loss += self.args.lambda_h * reg_terms.get("l0_h", 0.0)
                    
            loss = task_loss + sib_loss
            

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = target.size(0)
            running_loss += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{(running_loss / max(1, n)):.4f}")

        return running_loss / max(1, n)
    

    @torch.no_grad()
    def _evaluate(self, split: str, measure_time: bool = False):
        self.model.eval()
        loader = self.loader_dict[split]
        
        preds, gts = [], []
        inference_times = []
        
        for batch in tqdm.tqdm(loader):
            batch = batch.to(self.device, non_blocking=True)
            
            if measure_time:
                warmup_out = self.model(batch, self.entity_table, apply_pruning=True)
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                out = self.model(batch, self.entity_table, apply_pruning=True)
                
                torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append((end_time - start_time))  
            else:
                out = self.model(batch, self.entity_table, apply_pruning=True)
            
            logits = out[0] if isinstance(out, tuple) else out
            target = self._extract_target(batch)             

            if self.task_type == TaskType.REGRESSION and self.clamp_min is not None:
                y = logits.squeeze(-1)
                y = torch.clamp(y, self.clamp_min, self.clamp_max)
                logits = y.unsqueeze(-1)

            preds.append(logits.detach().cpu())
            gts.append(target.detach().cpu())

        y_pred = torch.cat(preds, dim=0)
        y_true = torch.cat(gts, dim=0)

        metric = self._compute_metric(y_pred, y_true)
        
        if measure_time and inference_times:
            avg_inference_time = np.sum(inference_times)
            std_inference_time = np.std(inference_times)
            return metric, avg_inference_time, std_inference_time
        
        return metric
    
    def test_with_time(self):
        if os.path.exists(self.best_path):
            self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        
        pruning_stats = self.model.get_pruning_stats()
        print("\n=== Pruning Statistics ===")
        print(f"Table-level decisions: {pruning_stats['table_decisions']}")
        for nt, col_info in pruning_stats['col_decisions'].items():
            print(f"Columns {nt}: {col_info['active']}/{col_info['total']} active ({col_info['ratio']:.1%})")
        
      
        print("\n=== Measuring Inference Time ===")
        test_metric, avg_time, std_time = self._evaluate(split="test", measure_time=True)
        print(f"[Test] {self.tune_metric}={test_metric:.4f}")
        print(f"[Time] Average inference time: {avg_time:.2f} ± {std_time:.2f} ms/batch")
        
        return test_metric, avg_time
    
    

    def _extract_target(self, batch):
        
        y = batch[self.entity_table].y

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.float().view(-1, 1)
        elif self.task_type == TaskType.REGRESSION:
            y = y.float().view(-1, 1)
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            y = y.float()
        return y
    

    def _compute_loss(self, logits, target):
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            return self.loss_fn(logits, target)
        elif self.task_type == TaskType.REGRESSION:
            return self.loss_fn(logits.squeeze(-1), target.squeeze(-1))
        else:
            raise RuntimeError("Unexpected task type")
        
    
    def normalize_task_type(self, tt, TaskType):
        if isinstance(tt, TaskType):
            return tt
        if isinstance(tt, Enum):
            return TaskType[tt.name]
        s = str(tt).strip()                        # e.g. "TaskType.BINARY_CLASSIFICATION"
        key = s.split(".")[-1].upper()            # "BINARY_CLASSIFICATION"
        return TaskType[key]



    def _init_task_cfg(self, task):
    
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            out_channels = 1
            loss_fn = BCEWithLogitsLoss()
            tune_metric = "roc_auc"
            higher_is_better = True
        elif self.task_type == TaskType.REGRESSION:
            out_channels = 1
            loss_fn = L1Loss()
            tune_metric = "mae"
            higher_is_better = False
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            out_channels = task.num_labels
            loss_fn = BCEWithLogitsLoss()
            tune_metric = "roc_auc"
            higher_is_better = True
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        return out_channels, loss_fn, tune_metric, higher_is_better
    

    def _compute_clamp_bounds_if_needed(self, task):
        if self.task_type != TaskType.REGRESSION:
            return None, None
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
        return float(clamp_min), float(clamp_max)
    

    def _compute_metric(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        t = self.task_type

        if t == TaskType.REGRESSION:
            mae = torch.mean(torch.abs(y_pred.squeeze(-1) - y_true.squeeze(-1))).item()
            return mae

        if t == TaskType.BINARY_CLASSIFICATION:
            # y_pred: [N,1]
            probs = torch.sigmoid(y_pred).cpu().numpy().ravel()
            labels = y_true.cpu().numpy().ravel().astype(np.int32)
            if _SK_AVAILABLE and len(np.unique(labels)) > 1:
                try:
                    return float(roc_auc_score(labels, probs))
                except Exception:
                    pass
            acc = (probs >= 0.5).astype(np.float32) == labels
            return float(acc.mean())

        if t == TaskType.MULTILABEL_CLASSIFICATION:
            # y_pred: [N, L]
            probs = torch.sigmoid(y_pred).cpu().numpy()
            labels = y_true.cpu().numpy().astype(np.int32)
            if _SK_AVAILABLE:
                ap_list = []
                for j in range(probs.shape[1]):
                    yj = labels[:, j]
                    pj = probs[:, j]
                    if len(np.unique(yj)) < 2:
                        continue
                    try:
                        ap_list.append(average_precision_score(yj, pj))
                    except Exception:
                        continue
                if len(ap_list) > 0:
                    return float(np.mean(ap_list))
            pred = (probs >= 0.5).astype(np.int32)
            eps = 1e-9
            f1s = []
            for j in range(pred.shape[1]):
                tp = np.sum((pred[:, j] == 1) & (labels[:, j] == 1))
                fp = np.sum((pred[:, j] == 1) & (labels[:, j] == 0))
                fn = np.sum((pred[:, j] == 0) & (labels[:, j] == 1))
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1 = 2 * precision * recall / (precision + recall + eps)
                f1s.append(f1)
            return float(np.mean(f1s))

        raise RuntimeError("Unexpected task type")
    


    