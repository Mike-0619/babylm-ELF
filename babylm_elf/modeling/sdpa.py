from __future__ import annotations

from contextlib import nullcontext
import os
import warnings

import torch
import torch.nn.functional as F


SDPA_BACKEND_ENV = "BABYLM_ELF_SDPA_BACKEND"
_VALID_BACKENDS = {"auto", "flash", "efficient", "math"}
_warned_flash_mask_fallback = False


def selected_sdpa_backend() -> str:
    backend = os.environ.get(SDPA_BACKEND_ENV, "auto").strip().lower() or "auto"
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{SDPA_BACKEND_ENV} must be one of auto, flash, efficient, math; "
            f"got {backend!r}."
        )
    return backend


def flash_attention_available() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    return bool(checker()) if checker is not None else True


def sdpa_backend_status() -> dict[str, str | bool | None]:
    cuda = torch.cuda.is_available()
    return {
        "env": SDPA_BACKEND_ENV,
        "selected": selected_sdpa_backend(),
        "flash_available": flash_attention_available(),
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled() if cuda else None,
        "efficient_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled() if cuda else None
        ),
        "math_enabled": torch.backends.cuda.math_sdp_enabled() if cuda else None,
    }


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    backend = selected_sdpa_backend()
    if backend == "flash" and attn_mask is not None:
        _warn_flash_mask_fallback()
        return _sdpa_attention_with_backend(
            "auto",
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    try:
        return _sdpa_attention_with_backend(
            backend,
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    except RuntimeError as exc:
        if backend == "auto":
            raise
        raise RuntimeError(
            _format_sdpa_failure(backend, query, key, value, attn_mask)
        ) from exc


def _sdpa_attention_with_backend(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    with _sdpa_kernel_context(backend):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )


def _warn_flash_mask_fallback() -> None:
    global _warned_flash_mask_fallback
    if _warned_flash_mask_fallback:
        return
    warnings.warn(
        f"{SDPA_BACKEND_ENV}=flash does not support explicit attention masks; "
        "using the fastest compatible SDPA backend for masked batches.",
        RuntimeWarning,
        stacklevel=2,
    )
    _warned_flash_mask_fallback = True


def _sdpa_kernel_context(backend: str):
    if backend == "auto":
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError(
            f"{SDPA_BACKEND_ENV}={backend!r} requires PyTorch SDPA backend "
            "selection support."
        ) from exc

    if backend == "flash":
        _require_flash_available()

    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel(backend_map[backend])


def _require_flash_available() -> None:
    if not flash_attention_available():
        raise RuntimeError(
            f"{SDPA_BACKEND_ENV}=flash requested, but PyTorch does not report "
            "FlashAttention as available on this process."
        )


def _format_sdpa_failure(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> str:
    return (
        f"SDPA backend {backend!r} failed for "
        f"q={tuple(query.shape)}, k={tuple(key.shape)}, v={tuple(value.shape)}, "
        f"mask={None if attn_mask is None else tuple(attn_mask.shape)}, "
        f"dtype={query.dtype}, device={query.device}."
    )
