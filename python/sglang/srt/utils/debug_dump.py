import os
import logging

_logger = logging.getLogger(__name__)

_DUMP_ENABLED: bool = os.environ.get("DUMP_TENSOR", "0") == "1"
_DUMP_DIR: str = os.environ.get("DUMP_TENSOR_DIR", "/mnt/raid0/zhaoan12/cache/qwen35_moe_tensor/sglang/")
_DUMP_LAYERS: set = (
    set(int(x) for x in os.environ.get("DUMP_TENSOR_LAYER", "").split(",") if x.strip())
    if os.environ.get("DUMP_TENSOR_LAYER", "")
    else set()
)
_DUMP_STEPS: set = (
    set(int(x) for x in os.environ.get("DUMP_TENSOR_STEPS", "0").split(",") if x.strip())
    if os.environ.get("DUMP_TENSOR_STEPS", "0")
    else {0}
)
_DUMP_RANK: int = int(os.environ.get("DUMP_TENSOR_RANK", "0"))
_dump_step_counter: int = 0
_rank_checked: bool = False
_rank_ok: bool = True


def dump_tensor_enabled() -> bool:
    return _DUMP_ENABLED


def dump_tensor_step_begin():
    pass


def dump_tensor_step_end():
    global _dump_step_counter
    if not _DUMP_ENABLED:
        return
    _dump_step_counter += 1


def _should_dump_layer(layer_idx: int) -> bool:
    if not _DUMP_LAYERS:
        return True
    return layer_idx in _DUMP_LAYERS


def _should_dump_step() -> bool:
    return _dump_step_counter in _DUMP_STEPS


def _check_rank() -> bool:
    global _rank_checked, _rank_ok
    if _rank_checked:
        return _rank_ok
    _rank_checked = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            _rank_ok = dist.get_rank() == _DUMP_RANK
        else:
            _rank_ok = True
    except Exception:
        _rank_ok = True
    if _rank_ok:
        _logger.info(f"[DUMP] Tensor dump active on rank {_DUMP_RANK}")
    return _rank_ok


def dump_tensor(
    tensor: "torch.Tensor",
    name: str,
    layer_idx: int = -1,
    save_file: bool = True,
):
    if not _DUMP_ENABLED:
        return
    if not _check_rank():
        return
    if not _should_dump_step():
        return
    if layer_idx >= 0 and not _should_dump_layer(layer_idx):
        return

    import torch

    t = tensor.detach().float()
    stats = (
        f"[DUMP] step={_dump_step_counter} {name}: "
        f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
        f"abs_mean={t.abs().mean().item():.6f}, "
        f"has_nan={tensor.isnan().any().item()}, "
        f"has_inf={tensor.isinf().any().item()}"
    )
    _logger.info(stats)
    print(stats, flush=True)

    if save_file:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        safe_name = name.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(_DUMP_DIR, f"step{_dump_step_counter}_{safe_name}.pt")
        torch.save(tensor.detach().cpu(), file_path)
