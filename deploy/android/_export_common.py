# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for exporting DreamLite modules to QNN-friendly ONNX.

This module is imported by the ``export_*_onnx.py`` scripts in this directory.
It centralises three concerns that are common to every export:

1. A QNN/HTP-friendly attention processor (``ExportFriendlyAttnProcessor``)
   that replaces ``F.scaled_dot_product_attention`` with an explicit
   matmul → (+mask) → softmax → matmul, and materialises the grouped-query
   (``kv_heads=1``) K/V to the full head count so the exported graph carries
   no head-dim broadcast.
2. A thin ``export_onnx`` wrapper around ``torch.onnx.export`` with sensible
   defaults (opset 17, static shapes, constant folding).
3. Post-processing helpers: ``simplify_onnx`` (onnx-simplifier),
   ``convert_fp16`` (onnxconverter-common) and ``check_parity`` which compares
   the PyTorch reference output against onnxruntime (CPU) within a tolerance.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# QNN-friendly attention processor
# =====================================================================
class ExportFriendlyAttnProcessor:
    """Drop-in replacement for ``AttnProcessor2_0`` used only during export.

    It reproduces the numerics of
    ``dreamlite.models.attention_processor.AttnProcessor2_0`` exactly, but:

    * expands grouped K/V (``attn.kv_heads`` may be 1) up to ``attn.heads``
      with an explicit ``repeat_interleave`` so the attention matmuls run at a
      single, static head count (QNN HTP dislikes head-dim broadcasting inside
      SDPA);
    * decomposes attention into ``matmul → scale → (+mask) → softmax → matmul``
      instead of calling ``F.scaled_dot_product_attention`` (which some ONNX /
      QNN converters lower poorly or not at all).
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # (batch, heads, query_len, key_len)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = attn.inner_dim
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.kv_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- Grouped-query expansion: broadcast kv_heads -> heads explicitly ---
        if attn.kv_heads != attn.heads:
            n_rep = attn.heads // attn.kv_heads
            # Keep the expansion entirely 4-D.  repeat_interleave is
            # numerically equivalent but the ONNX exporter lowers it to
            # Unsqueeze(5-D) -> Tile -> Reshape, which is rejected by some
            # QNN converters.  repeat emits a direct 4-D Tile instead.
            key = key.repeat(1, n_rep, 1, 1)
            value = value.repeat(1, n_rep, 1, 1)

        # --- Explicit scaled dot-product attention (no SDPA fusion) ---
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attn_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def apply_export_friendly_attention(unet) -> None:
    """Replace every attention processor on ``unet`` with the export-friendly one."""
    unet.set_attn_processor(ExportFriendlyAttnProcessor())


# =====================================================================
# ONNX export + post-processing
# =====================================================================
def export_onnx(
    module: torch.nn.Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    save_path: Path,
    opset: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> None:
    """Trace ``module`` and export to ONNX with QNN-friendly defaults.

    Shapes are static by default (``dynamic_axes=None``) because QNN HTP wants
    fixed dimensions. Constant folding is enabled to bake the traced constants.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            str(save_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    print(f"  [onnx] exported -> {save_path}")


def simplify_onnx(save_path: Path) -> None:
    """Run onnx-simplifier in place; skip gracefully if unavailable."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  [simplify] onnxsim not installed, skipping.")
        return

    model = onnx.load(str(save_path))
    model_simp, ok = simplify(model)
    if ok:
        onnx.save(model_simp, str(save_path))
        print(f"  [simplify] simplified -> {save_path}")
    else:
        print("  [simplify] simplification check failed, kept original.")


def convert_fp16(save_path: Path, fp16_path: Path) -> None:
    """Convert an FP32 ONNX model to FP16 (new Snapdragon HTP runs FP16 natively)."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("  [fp16] onnxconverter-common not installed, skipping FP16 export.")
        return

    model = onnx.load(str(save_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(fp16_path))
    print(f"  [fp16] exported -> {fp16_path}")


def check_onnx(save_path: Path) -> None:
    """Run onnx.checker to validate graph well-formedness."""
    try:
        import onnx
    except ImportError:
        print("  [check] onnx not installed, skipping checker.")
        return
    model = onnx.load(str(save_path))
    onnx.checker.check_model(model)
    print("  [check] onnx.checker passed.")


def check_parity(
    save_path: Path,
    torch_output: torch.Tensor,
    ort_feeds: Dict[str, np.ndarray],
    output_name: str,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> None:
    """Compare PyTorch reference output vs onnxruntime (CPU) output.

    Prints max absolute / relative error and whether it is within tolerance.
    Tolerances are loose enough for FP32 tracing noise but will catch real
    numerical divergence from the attention rewrite.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [parity] onnxruntime not installed, skipping parity check.")
        return

    sess = ort.InferenceSession(str(save_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run([output_name], ort_feeds)[0]

    ref = torch_output.detach().cpu().numpy().astype(np.float32)
    ort_out = ort_out.astype(np.float32)

    abs_diff = np.abs(ref - ort_out)
    denom = np.abs(ref) + 1e-6
    rel_diff = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    within = bool(np.allclose(ref, ort_out, rtol=rtol, atol=atol))

    print(f"  [parity] max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  "
          f"within(rtol={rtol}, atol={atol})={within}")
    if not within:
        print("  [parity] ⚠️  output exceeds tolerance — inspect the export before trusting it.")


def to_numpy_feeds(names: Sequence[str], tensors: Sequence[torch.Tensor]) -> Dict[str, np.ndarray]:
    """Build an onnxruntime feed dict from parallel names/tensors lists."""
    return {n: t.detach().cpu().numpy() for n, t in zip(names, tensors)}


def report_size(save_path: Path) -> None:
    """Print the on-disk size of an exported ONNX model (file or dir)."""
    p = Path(save_path)
    if p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        size = p.stat().st_size
    print(f"  [size] {p.name}: {size / 1024**2:.1f} MB")
