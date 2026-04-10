#!/usr/bin/env python3
"""Simple tensor comparer for RTP vs SGLang MoE dumps."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

DEFAULT_RTP_DIR = "/mnt/raid0/yilin/moe_dump/"
DEFAULT_SGLANG_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/sglang_fused_moe"


@dataclass
class Result:
    name: str
    status: str
    note: str
    cosine: Optional[float] = None
    diff_min: Optional[float] = None
    diff_median: Optional[float] = None
    diff_max: Optional[float] = None
    diff_mean: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare .pt tensors from two directories.")
    parser.add_argument("--rtp-dir", default=DEFAULT_RTP_DIR)
    parser.add_argument("--sglang-dir", default=DEFAULT_SGLANG_DIR)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--report-file", default="tensor_compare_report.txt")
    return parser.parse_args()


def normalize_name(file_name: str) -> str:
    # RTP often uses *_scale1.pt while SGLang uses *_scale.pt
    if file_name.endswith("_scale1.pt"):
        return file_name.replace("_scale1.pt", "_scale.pt")
    return file_name


def list_tensors(directory: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(directory.glob("*.pt")):
        out[normalize_name(p.name)] = p
    return out


def load_tensor(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not torch.is_tensor(obj):
        raise TypeError(f"Not a tensor: {path}")
    return obj.detach().cpu()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(1, -1).float()
    b = b.reshape(1, -1).float()
    an = a.norm(p=2).item()
    bn = b.norm(p=2).item()
    if math.isclose(an, 0.0) and math.isclose(bn, 0.0):
        return 1.0
    if math.isclose(an, 0.0) or math.isclose(bn, 0.0):
        return 0.0
    return torch.nn.functional.cosine_similarity(a, b).item()


def compare_pair(name: str, rtp: torch.Tensor, sgl: torch.Tensor, atol: float, rtol: float) -> Result:
    note = "same_shape"
    if rtp.shape != sgl.shape:
        if rtp.numel() != sgl.numel():
            return Result(
                name=name,
                status="FAIL",
                note=f"shape mismatch: RTP{list(rtp.shape)} vs SGL{list(sgl.shape)}",
            )
        rtp = rtp.reshape(sgl.shape)
        note = f"reshape RTP{list(rtp.shape)} -> SGL{list(sgl.shape)}"

    diff = (rtp.float() - sgl.float()).abs().reshape(-1)
    ok = torch.allclose(rtp.float(), sgl.float(), atol=atol, rtol=rtol)
    return Result(
        name=name,
        status="PASS" if ok else "FAIL",
        note=note if ok else f"{note}; allclose failed",
        cosine=cosine_similarity(rtp, sgl),
        diff_min=diff.min().item(),
        diff_median=diff.median().item(),
        diff_max=diff.max().item(),
        diff_mean=diff.mean().item(),
    )


def format_report(results: List[Result], atol: float, rtol: float) -> str:
    lines: List[str] = []
    lines.append(f"Tolerance: atol={atol}, rtol={rtol}")
    lines.append("-" * 120)
    lines.append(
        f"{'Tensor':35} {'Status':6} {'Cosine':>9} {'Diff[min|med|max|mean]':>42}  Note"
    )
    lines.append("-" * 120)
    for r in results:
        if r.cosine is None:
            diff_str = "-"
            cos_str = "-"
        else:
            diff_str = (
                f"{r.diff_min:.3e}|{r.diff_median:.3e}|{r.diff_max:.3e}|{r.diff_mean:.3e}"
            )
            cos_str = f"{r.cosine:.6f}"
        lines.append(f"{r.name:35} {r.status:6} {cos_str:>9} {diff_str:>42}  {r.note}")

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    lines.append("-" * 120)
    lines.append(f"Summary: total={total}, pass={passed}, fail={failed}, skip={skipped}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rtp_dir = Path(args.rtp_dir)
    sgl_dir = Path(args.sglang_dir)

    if not rtp_dir.is_dir():
        print(f"RTP directory not found: {rtp_dir}")
        return 1
    if not sgl_dir.is_dir():
        print(f"SGLang directory not found: {sgl_dir}")
        return 1

    rtp_files = list_tensors(rtp_dir)
    sgl_files = list_tensors(sgl_dir)
    names = sorted(set(rtp_files.keys()) | set(sgl_files.keys()))

    results: List[Result] = []
    for name in names:
        rp = rtp_files.get(name)
        sp = sgl_files.get(name)
        if rp is None:
            results.append(Result(name, "SKIP", "Only in SGLang"))
            continue
        if sp is None:
            results.append(Result(name, "SKIP", "Only in RTP"))
            continue

        try:
            rt = load_tensor(rp)
            st = load_tensor(sp)
            results.append(compare_pair(name, rt, st, args.atol, args.rtol))
        except Exception as e:
            results.append(Result(name, "FAIL", f"load/compare error: {e}"))

    report = format_report(results, args.atol, args.rtol)
    print(report)
    Path(args.report_file).write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {args.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
