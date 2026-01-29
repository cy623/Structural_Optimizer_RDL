import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import warnings


class StructureGate(nn.Module):
    def __init__(self,
                 node_types: List[str],
                 num_features_dict: Dict[str, int],
                 template_names: List[str] = None,
                 temp: float = 1.0,
                 init_log_alpha: float = 0.0,
                 column_len_policy: str = "auto_expand"):
        super().__init__()
        assert column_len_policy in {"auto_expand", "clip", "raise"}
        self.temp = float(temp)
        self.node_types = list(node_types)
        self.column_len_policy = column_len_policy

        self.table_log_alpha = nn.ParameterDict({
            nt: nn.Parameter(torch.tensor(float(init_log_alpha))) for nt in self.node_types
        })

        self.column_log_alpha = nn.ParameterDict({
            nt: nn.Parameter(torch.zeros(int(num_features_dict[nt]))) for nt in self.node_types
        })

        self.template_names = [str(n) for n in (template_names or [])]
        self.aug_log_alpha = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_alpha))) for name in self.template_names
        })


        self._warned_ntypes = set()

    @staticmethod
    def _bern_gumbel_scalar(logit: torch.Tensor, tau: float, hard: bool, training: bool):
        if training:
            logits = torch.stack([torch.zeros_like(logit), logit], dim=-1)   # [2]
            y = F.gumbel_softmax(logits, tau=tau, hard=hard)[..., 1]         
        else:
            y = (logit > 0).to(logit.dtype)
        p = torch.sigmoid(logit)
        return y, p

    @staticmethod
    def _bern_gumbel_vec(logits_vec: torch.Tensor, tau: float, hard: bool, training: bool):
        if training:
            logits = torch.stack([torch.zeros_like(logits_vec), logits_vec], dim=-1)  # [F,2]
            y = F.gumbel_softmax(logits, tau=tau, hard=hard)[..., 1]                  # [F]
        else:
            y = (logits_vec > 0).to(logits_vec.dtype)
        p = torch.sigmoid(logits_vec)
        return y, p

    def _ensure_column_len(self, nt: str, F: int) -> int:
        param: torch.Tensor = self.column_log_alpha[nt]
        P = param.numel()

        if F == P:
            return F

        policy = self.column_len_policy
        if policy == "raise":
            raise ValueError(f"[StructureGate] Column size mismatch for '{nt}': input F={F}, param={P}")

        if policy == "clip":
            if nt not in self._warned_ntypes:
                warnings.warn(f"[StructureGate] '{nt}' uses clip policy: input F={F}, param={P}. "
                              f"Using first L={min(F, P)} columns; others are not gated/regularized.",
                              stacklevel=2)
                self._warned_ntypes.add(nt)
            return min(F, P)

        if F > P:
            device, dtype = param.device, param.dtype
            new_param = torch.zeros(F, device=device, dtype=dtype)
            if P > 0:
                new_param[:P] = param.data
            self.column_log_alpha[nt] = nn.Parameter(new_param) 
            if nt not in self._warned_ntypes:
                warnings.warn(f"[StructureGate] '{nt}' column params auto-expanded: {P} -> {F}.",
                              stacklevel=2)
                self._warned_ntypes.add(nt)
            return F
        else:  # F < P
            if nt not in self._warned_ntypes:
                warnings.warn(f"[StructureGate] '{nt}' input has fewer cols than params: F={F} < {P}. "
                              f"Using first L={F}. Consider shrinking params offline if persistent.",
                              stacklevel=2)
                self._warned_ntypes.add(nt)
            return F

    def forward(self,
            x_raw_dict: Dict[str, torch.Tensor],
            *,
            training: bool = True,
            hard: bool = True):
  
        device = next(self.parameters()).device

        reg_table = torch.tensor(0.0, device=device)
        reg_column = torch.tensor(0.0, device=device)
        reg_augment = torch.tensor(0.0, device=device)

        x_scaled: Dict[str, torch.Tensor] = {}
        table_gates_smpl, column_gates_smpl, aug_gates_smpl = {}, {}, {}

        for nt in self.node_types:
            if nt not in x_raw_dict:
                continue
            x = x_raw_dict[nt]
            assert x.dim() == 2, f"x_raw_dict['{nt}'] must be 2D, got {tuple(x.shape)}"

            col_param = self.column_log_alpha[nt]            # [F]
            F_in, F_param = x.size(1), col_param.numel()
            if F_in != F_param:
                raise ValueError(
                    f"[StructureGate] Column size mismatch for '{nt}': "
                    f"input F={F_in}, param F={F_param}. "
                    f"Ensure your upstream feature extraction matches num_features_dict."
                )

            g_s, g_p = self._bern_gumbel_scalar(self.table_log_alpha[nt], self.temp, hard, training)

            w_s, w_p = self._bern_gumbel_vec(col_param, self.temp, hard, training)  # [F]
            mask = w_s.to(dtype=x.dtype, device=x.device)                            # [F]
            x_scaled[nt] = x * mask                                                  # [N,F] * [F] -> [N,F]

            reg_table = reg_table + g_p
            reg_column = reg_column + w_p.sum()

            table_gates_smpl[nt] = g_s
            column_gates_smpl[nt] = w_s

        for name in self.template_names:
            h_s, h_p = self._bern_gumbel_scalar(self.aug_log_alpha[name], self.temp, hard, training)
            aug_gates_smpl[name] = h_s
            reg_augment = reg_augment + h_p

        regs = {'table': reg_table, 'column': reg_column, 'augment': reg_augment}
        gates = {'table': table_gates_smpl, 'column': column_gates_smpl, 'augment': aug_gates_smpl}
        return x_scaled, regs, gates

    def expected_gate_values(self):
        table = {nt: torch.sigmoid(self.table_log_alpha[nt]) for nt in self.node_types}
        column = {nt: torch.sigmoid(self.column_log_alpha[nt]) for nt in self.node_types}
        augment = {name: torch.sigmoid(self.aug_log_alpha[name]) for name in self.template_names}
        return table, column, augment
    
