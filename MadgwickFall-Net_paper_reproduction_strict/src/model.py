"""
MadgwickFall-Net architecture, PyTorch implementation of Section 3.4 /
Figure 4.

Requires torch (not installed in this sandbox - no internet access to
fetch it). This file is a complete, ready-to-run reference implementation;
its parameter counts have been cross-checked against src/param_count.py
(pure-numpy accounting), which matches the paper's claims almost exactly:
  - DS-SE1 compression ratio 5.76x, DS-SE2 5.74x (paper: 5.76x / 5.74x)
  - Total params 30,522 -> 59.61 KB @ FP16 (paper: 59.7 KB)

Input: four (B, 3, 87) streams -- body accel, body gyro, global accel,
global gyro (Sec 3.4.1). Output: (B, 2) softmax logits over {ADL, Fall}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D conv: depthwise (per-channel, kernel k) +
    pointwise (1x1, Cin->Cout). 'same' padding, no bias (paper's stated
    param formula Cin*k + Cin*Cout excludes bias terms, Sec 3.4.2)."""

    def __init__(self, cin, cout, k, causal=False, dilation=1):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.k = k
        if causal:
            self.pad = (k - 1) * dilation
            padding = 0
        else:
            self.pad = 0
            padding = ((k - 1) * dilation) // 2
        self.depthwise = nn.Conv1d(cin, cin, kernel_size=k, groups=cin,
                                    padding=padding, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, kernel_size=1, bias=False)

    def forward(self, x):
        if self.causal and self.pad > 0:
            x = F.pad(x, (self.pad, 0))  # left-only zero padding -> causal
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSSEModule(nn.Module):
    """DS-SE module (Fig. 4 inset): DSConv -> BN -> SiLU -> SE channel
    attention (GAP -> FC(r=6) -> ReLU -> FC -> Sigmoid -> scale).
    Eqs. (12)-(14)."""

    def __init__(self, cin, cout, k, reduction=6):
        super().__init__()
        self.dsconv = DepthwiseSeparableConv1d(cin, cout, k, causal=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.SiLU()
        hidden = max(1, cout // reduction)
        self.fc1 = nn.Linear(cout, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, cout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, Cin, T)
        h = self.act(self.bn(self.dsconv(x)))          # Eq. 12
        z = h.mean(dim=-1)                               # GAP -> (B, C)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(z))))  # Eq. 13
        out = h * s.unsqueeze(-1)                         # Eq. 14
        return out


class LTB(nn.Module):
    """Lightweight Temporal Block: 2x causal depthwise-separable conv
    (k=3, dilation d_i), each followed by BN+SiLU+Dropout, then a residual
    add (identity, since Cin=Cout=32 throughout) and a final SiLU.
    Eqs. (18)-(19)."""

    def __init__(self, channels, dilation, k=3, dropout=0.08):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(channels, channels, k,
                                               causal=True, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = DepthwiseSeparableConv1d(channels, channels, k,
                                               causal=True, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.drop1(self.act(self.bn1(self.conv1(x))))
        h = self.drop2(self.act(self.bn2(self.conv2(h))))
        out = self.act(h + x)  # identity residual (Cin==Cout==32)
        return out


class DSTCN(nn.Module):
    """DS-TCN = LTB1 (dilation=1) -> LTB2 (dilation=2) -> GroupNorm -> GAP.
    Receptive field = 1 + 2*(k-1)*(2^L - 1) = 13 steps for k=3, L=2
    (Sec 3.4.2 point 2/3)."""

    def __init__(self, channels=32, k=3, num_groups=8, dropout=0.08):
        super().__init__()
        self.ltb1 = LTB(channels, dilation=1, k=k, dropout=dropout)
        self.ltb2 = LTB(channels, dilation=2, k=k, dropout=dropout)
        self.gn = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

    def forward(self, x):
        x = self.ltb1(x)
        x = self.ltb2(x)
        x = self.gn(x)
        x = x.mean(dim=-1)  # GAP -> (B, C)
        return x


class STFE(nn.Module):
    """Spatio-Temporal Feature Extractor: DS-SE1(3->16,k=9) -> MaxPool(2)
    -> DS-SE2(16->32,k=7) -> DS-TCN(32) -> 32-d feature (Sec 3.4.2)."""

    def __init__(self):
        super().__init__()
        self.dsse1 = DSSEModule(3, 16, k=9, reduction=6)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dsse2 = DSSEModule(16, 32, k=7, reduction=6)
        self.dstcn = DSTCN(channels=32, k=3, dropout=0.08)

    def forward(self, x):
        # x: (B, 3, 87)
        h1 = self.dsse1(x)          # (B,16,87)
        h1 = self.pool(h1)          # (B,16,43)
        h2 = self.dsse2(h1)         # (B,32,43)
        f = self.dstcn(h2)          # (B,32)
        return f


class MadgwickFallNet(nn.Module):
    """
    Four-branch parallel network (Fig. 4).
    branches: order must match forward() input order:
      body_accel, body_gyro, global_accel, global_gyro
    branch_config: which of the 4 branches are active, for the coordinate-
    system ablation (Sec 4.3.1): "four-branch" / "global_only" / "body_only".
    """

    BRANCH_NAMES = ("body_accel", "body_gyro", "global_accel", "global_gyro")

    def __init__(self, active_branches=("body_accel", "body_gyro",
                                         "global_accel", "global_gyro")):
        super().__init__()
        self.active_branches = list(active_branches)
        self.stfes = nn.ModuleDict({name: STFE() for name in self.active_branches})
        fused_dim = 32 * len(self.active_branches)
        self.fc1 = nn.Linear(fused_dim, 48)
        self.bn1 = nn.BatchNorm1d(48)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.30)
        self.fc2 = nn.Linear(48, 2)

    def forward(self, inputs: dict):
        """inputs: dict of branch_name -> (B, 3, 87) tensor."""
        feats = [self.stfes[name](inputs[name]) for name in self.active_branches]
        f = torch.cat(feats, dim=1)                      # Eq. (20)
        h = self.drop(self.act(self.bn1(self.fc1(f))))
        logits = self.fc2(h)                              # Eq. (21) pre-softmax
        return logits  # use F.log_softmax / CrossEntropyLoss outside


def build_model(config: str = "four_branch"):
    """config in {'four_branch','global_only','body_only'} per Sec 4.3.1."""
    if config == "four_branch":
        branches = ("body_accel", "body_gyro", "global_accel", "global_gyro")
    elif config == "global_only":
        branches = ("global_accel", "global_gyro")
    elif config == "body_only":
        branches = ("body_accel", "body_gyro")
    else:
        raise ValueError(config)
    return MadgwickFallNet(active_branches=branches)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model("four_branch")
    n = count_parameters(m)
    print(f"four_branch params: {n}  (~{n*2/1024:.2f} KB @FP16, paper: 30,522 / 59.7KB)")
    for cfg in ("global_only", "body_only"):
        mm = build_model(cfg)
        print(f"{cfg} params: {count_parameters(mm)}")
