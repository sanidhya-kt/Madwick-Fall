"""
Structural / parameter-count sanity check for MadgwickFall-Net, matching
Section 3.4 and Figure 4 of the paper, without requiring PyTorch (not
installed in this sandbox / no internet to fetch it).

Verifies:
  - DS-SE1 / DS-SE2 depthwise-separable compression ratios against the
    paper's stated 5.76x and 5.74x (Sec. 3.4.2).
  - Total parameter count against the paper's "59.7 KB" claim, which
    Sec. 4.4.3 clarifies is FP16-quantized weight storage
    ("trained model weights are saved in FP16 format (size: 59.7 KB)").
    So 59.7 KB / 2 bytes ~= 30,566 params expected.
"""
from dataclasses import dataclass


def ds_conv_params(cin, cout, k, bias=False):
    """Depthwise separable conv = depthwise (Cin*k[*1 if bias]) + pointwise
    (Cin*Cout[+Cout if bias]). Paper's own formula (Sec 3.4.2):
    'compressed from Cin×Cout×k to Cin×k + Cin×Cout' -- no bias terms
    mentioned, so bias=False reproduces the paper's stated ratios exactly."""
    depthwise = cin * k + (cin if bias else 0)
    pointwise = cin * cout + (cout if bias else 0)
    return depthwise + pointwise


def standard_conv_params(cin, cout, k):
    return cin * cout * k


def bn_params(c):
    return 2 * c  # gamma, beta (trainable only; running stats not counted)


def se_params(c, r=6, bias=True):
    """Squeeze-Excite: two FC layers, reduction ratio r (paper Sec 3.4.2)."""
    hidden = max(1, c // r)
    w1 = c * hidden + (hidden if bias else 0)
    w2 = hidden * c + (c if bias else 0)
    return w1 + w2


def ds_se_module_params(cin, cout, k, r=6):
    conv = ds_conv_params(cin, cout, k)
    bn = bn_params(cout)
    se = se_params(cout, r=r)
    total = conv + bn + se
    std = standard_conv_params(cin, cout, k)
    ratio = std / conv
    return total, ratio


def cdsc_params(c, k=3):
    """Causal depthwise separable conv, Cin=Cout=c (paper: DS-TCN channels
    fixed at 32, Sec 3.4.2 point 3)."""
    return ds_conv_params(c, c, k)


def ltb_params(c, k=3, use_residual_proj=False):
    """Lightweight Temporal Block: 2x CDSC + 2x BN (+ optional 1x1 conv
    residual projection, not needed here since Cin=Cout=32 throughout,
    per paper: 'residual path directly performs identity mapping')."""
    conv = 2 * cdsc_params(c, k)
    bn = 2 * bn_params(c)
    res = standard_conv_params(c, c, 1) if use_residual_proj else 0
    return conv + bn + res


def ds_tcn_params(c=32, n_ltb=2, k=3, group_norm=True):
    ltb_total = n_ltb * ltb_params(c, k)
    gn = bn_params(c) if group_norm else 0  # GroupNorm affine: 2*C
    return ltb_total + gn


def stfe_params():
    """One Spatio-Temporal Feature Extractor branch:
       DS-SE1 (3->16, k=9) -> MaxPool -> DS-SE2 (16->32, k=7) -> DS-TCN(32)
       -> (GAP has no params) -> GroupNorm(32)."""
    se1, ratio1 = ds_se_module_params(3, 16, 9, r=6)
    se2, ratio2 = ds_se_module_params(16, 32, 7, r=6)
    tcn = ds_tcn_params(32, n_ltb=2, k=3, group_norm=False)  # GN counted below once
    final_gn = bn_params(32)
    total = se1 + se2 + tcn + final_gn
    return total, ratio1, ratio2


def classifier_params(in_dim=128, hidden=48, out_dim=2, bias=True):
    fc1 = in_dim * hidden + (hidden if bias else 0)
    bn1 = bn_params(hidden)
    fc2 = hidden * out_dim + (out_dim if bias else 0)
    return fc1 + bn1 + fc2


def full_model_params(n_branches=4):
    stfe_total, r1, r2 = stfe_params()
    branches = n_branches * stfe_total
    clf = classifier_params()
    total = branches + clf
    return {
        "stfe_params_per_branch": stfe_total,
        "n_branches": n_branches,
        "branches_total": branches,
        "classifier_total": clf,
        "total_params": total,
        "ds_se1_compression_ratio": r1,
        "ds_se2_compression_ratio": r2,
    }


if __name__ == "__main__":
    result = full_model_params()
    for k, v in result.items():
        print(f"{k:32s}: {v}")

    total = result["total_params"]
    fp16_kb = total * 2 / 1024
    fp32_kb = total * 4 / 1024
    print(f"\nEstimated size @ FP16: {fp16_kb:.2f} KB  (paper claims 59.7 KB)")
    print(f"Estimated size @ FP32: {fp32_kb:.2f} KB")
    print(f"\nDS-SE1 compression ratio: {result['ds_se1_compression_ratio']:.2f}x "
          f"(paper states ~5.76x)")
    print(f"DS-SE2 compression ratio: {result['ds_se2_compression_ratio']:.2f}x "
          f"(paper states ~5.74x)")
