#!/usr/bin/env python3
"""Compare tensor dumps between SGLang and RTP-LLM."""

import os
import torch
# SGLANG_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/sglang_perblock_fp8/"
# RTP_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/rtp_perblock_4_layers/"
# RTP_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/rtp_perblock_4_layers_0331_1445/"



SGLANG_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/sglang_ptpc/"
RTP_DIR = "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/rtp_ptpc_0325_4_layers/"

# Qwen3.5-397B full attention dims (per TP rank, TP=8)
# num_attention_heads=32, num_key_value_heads=2, head_dim=256
# q_local=32/8*256=1024, kv_local_sglang=max(1,2/8)*256=256
# RTP duplicates kv heads: kv_local_rtp=2*256=512
FULL_ATTN_Q_DIM = 1024
FULL_ATTN_KV_DIM_SGLANG = 256
FULL_ATTN_KV_DIM_RTP = 512

# Qwen3.5 推理顺序
ORDER = [
    "embedding_out",
    # per-layer pattern
    "layer{i}.input",
    "layer{i}.input_layernorm_out",
    # linear attn layers
    "layer{i}.linear_attn.mixed_qkv",
    "layer{i}.linear_attn.z",
    "layer{i}.linear_attn.b",
    "layer{i}.linear_attn.a",
    "layer{i}.linear_attn.fla_out",
    "layer{i}.linear_attn.norm_out",
    "layer{i}.linear_attn.out_proj",
    "layer{i}.linear_attn.all_reduce",
    # full attn layers
    "layer{i}.full_attn.qkv_proj_out",
    "layer{i}.full_attn.gate",
    "layer{i}.full_attn.qk_norm_out",
    "layer{i}.full_attn.fmha_out",
    "layer{i}.full_attn.o_proj_out",
    "layer{i}.full_attn.allreduce_out",
    "layer{i}.full_attn.out",
    # attn output
    "layer{i}.attn_out",
    "layer{i}.post_attn_layernorm_out",
    # moe
    "layer{i}.moe.router_logits",
    "layer{i}.moe.topk_weights",
    "layer{i}.moe.topk_ids",
    "layer{i}.moe.experts_out",
    "layer{i}.moe.shared_expert_out",
    "layer{i}.moe.shared_expert_gated",
    "layer{i}.moe.out",
    # mlp output
    "layer{i}.mlp_out",
    # final
    "final_norm_out",
]

ALIASES = {
    # allow historical naming variants
    "full_attn.allreduce_out": "full_attn.all_reduce",
}


def normalize_tensor_name(name: str) -> str:
    for src, dst in ALIASES.items():
        name = name.replace(src, dst)
    return name


def sorted_by_inference_order(names, num_layers=5):
    ordered = []
    for pat in ORDER:
        if "{i}" in pat:
            for i in range(num_layers):
                n = pat.replace("{i}", str(i)) + ".pt"
                if n in names:
                    ordered.append(n)
        else:
            n = pat + ".pt"
            if n in names:
                ordered.append(n)
    for n in sorted(names):
        if n not in ordered:
            ordered.append(n)
    return ordered


def compare_one(f1, f2, tag, colors):
    """Compare two float tensors, return (ok, msg)."""
    G, R, D, RST = colors
    cos = torch.nn.functional.cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1)).item()
    max_diff = (f1 - f2).abs().max().item()
    ok = torch.allclose(f1, f2, atol=1e-2, rtol=1e-1)
    if ok:
        return True, f"  {G}PASS{RST}  {tag}  {D}cos={cos:.6f}  max_diff={max_diff:.6f}{RST}"
    else:
        return False, f"  {R}FAIL{RST}  {tag}  cos={cos:.6f}  max_diff={max_diff:.6f}"


def split_qkv_and_compare(t_sglang, t_rtp, tag, colors):
    """Split full_attn qkv/qk tensors by q/k/v dims, handle RTP kv duplication."""
    results = []
    s, r = t_sglang.float(), t_rtp.float()
    is_qk_only = "qk_norm_out" in tag

    if is_qk_only:
        # SGLang: [q, k], RTP: [q, k_dup, v] (qk_fuse_norm output includes v)
        q_s = s[:, :FULL_ATTN_Q_DIM]
        k_s = s[:, FULL_ATTN_Q_DIM:]
        q_r = r[:, :FULL_ATTN_Q_DIM]
        k_r = r[:, FULL_ATTN_Q_DIM:FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_RTP]
        k_r = k_r[:, :FULL_ATTN_KV_DIM_SGLANG]
        results.append(compare_one(q_s, q_r, f"{tag}[q]", colors))
        results.append(compare_one(k_s, k_r, f"{tag}[k]", colors))
    else:
        # SGLang: [q, k, v], RTP: [q, k_dup, v_dup]
        q_s = s[:, :FULL_ATTN_Q_DIM]
        k_s = s[:, FULL_ATTN_Q_DIM:FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_SGLANG]
        v_s = s[:, FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_SGLANG:]

        q_r = r[:, :FULL_ATTN_Q_DIM]
        k_r = r[:, FULL_ATTN_Q_DIM:FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_SGLANG]
        v_r = r[:, FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_RTP:
                   FULL_ATTN_Q_DIM + FULL_ATTN_KV_DIM_RTP + FULL_ATTN_KV_DIM_SGLANG]

        results.append(compare_one(q_s, q_r, f"{tag}[q]", colors))
        results.append(compare_one(k_s, k_r, f"{tag}[k]", colors))
        results.append(compare_one(v_s, v_r, f"{tag}[v]", colors))

    return results


def discover_steps(directory):
    """Find all step IDs from dump files."""
    import re
    steps = set()
    for f in os.listdir(directory):
        m = re.match(r"step(\d+)_", f)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def compare_step(step_id, colors):
    """Compare all tensors for a given step. Returns (total, passed, failed, skipped)."""
    G, R, Y, D, RST = colors
    step_prefix = f"step{step_id}_"

    sglang_files = {
        normalize_tensor_name(f.replace(step_prefix, "")): f
        for f in os.listdir(SGLANG_DIR)
        if f.startswith(step_prefix) and f.endswith(".pt")
    }
    rtp_files = {
        normalize_tensor_name(f.replace(step_prefix, "")): f
        for f in os.listdir(RTP_DIR)
        if f.startswith(step_prefix) and f.endswith(".pt")
    }

    all_names = sorted(set(sglang_files.keys()) | set(rtp_files.keys()))
    ordered = sorted_by_inference_order(all_names)

    total = passed = failed = skipped = 0

    for name in ordered:
        tag = name.replace(".pt", "")

        if name not in sglang_files:
            print(f"  {Y}SKIP{RST}  {tag}  (RTP only)")
            skipped += 1
            continue
        if name not in rtp_files:
            print(f"  {Y}SKIP{RST}  {tag}  (SGLang only)")
            skipped += 1
            continue

        t1 = torch.load(os.path.join(SGLANG_DIR, sglang_files[name]), map_location="cpu", weights_only=True)
        t2 = torch.load(os.path.join(RTP_DIR, rtp_files[name]), map_location="cpu", weights_only=True)

        if ("full_attn.qkv_proj_out" in tag or "full_attn.qk_norm_out" in tag) and t1.shape != t2.shape:
            results = split_qkv_and_compare(t1, t2, tag, (G, R, D, RST))
            for ok, msg in results:
                total += 1
                print(msg)
                if ok:
                    passed += 1
                else:
                    failed += 1
            continue

        total += 1

        if t1.shape != t2.shape:
            if t1.numel() == t2.numel():
                t1 = t1.reshape(t2.shape)
            else:
                print(f"  {R}FAIL{RST}  {tag}  shape+numel mismatch: sglang={list(t1.shape)} rtp={list(t2.shape)}")
                failed += 1
                continue

        ok, msg = compare_one(t1.float(), t2.float(), tag, (G, R, D, RST))
        print(msg)
        if ok:
            passed += 1
        else:
            failed += 1

    return total, passed, failed, skipped


def main():
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    D = "\033[2m"
    B = "\033[1m"
    RST = "\033[0m"
    colors = (G, R, Y, D, RST)

    sglang_steps = discover_steps(SGLANG_DIR)
    rtp_steps = discover_steps(RTP_DIR)
    common_steps = sorted(set(sglang_steps) & set(rtp_steps))

    if not common_steps:
        print(f"  {R}No common steps found.{RST}")
        print(f"  SGLang steps: {sglang_steps}")
        print(f"  RTP steps:    {rtp_steps}")
        return

    grand_total = grand_passed = grand_failed = grand_skipped = 0
    step_labels = {0: "prefill", 1: "decode-1"}

    for step_id in common_steps:
        label = step_labels.get(step_id, f"step-{step_id}")
        print(f"\n{B}{'='*60}")
        print(f"  Step {step_id} ({label})")
        print(f"{'='*60}{RST}")

        total, passed, failed, skipped = compare_step(step_id, colors)
        print(f"\n  Step {step_id}: Total={total}  {G}PASS={passed}{RST}  {R}FAIL={failed}{RST}  SKIP={skipped}")

        grand_total += total
        grand_passed += passed
        grand_failed += failed
        grand_skipped += skipped

    if len(common_steps) > 1:
        print(f"\n{B}{'='*60}")
        print(f"  Overall Summary")
        print(f"{'='*60}{RST}")
        print(f"  Total={grand_total}  {G}PASS={grand_passed}{RST}  {R}FAIL={grand_failed}{RST}  SKIP={grand_skipped}")


if __name__ == "__main__":
    main()
