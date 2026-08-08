# DreamLite Android Deployment Guide (ONNX + Qualcomm QNN)

This directory contains the **model export** code needed to deploy DreamLite on
Android via **ONNX Runtime** with the **Qualcomm QNN / Hexagon NPU** backend.

It is the Android counterpart to the iOS deployment in [`../`](../README.md)
(CoreML + MLX). The runtime logic (scheduler, prompt templates, in-context
concatenation) is identical to iOS — see [`../DreamLitePipeline.swift`](../DreamLitePipeline.swift)
and [`../FluxScheduler.swift`](../FluxScheduler.swift) as the reference to port
to Kotlin/C++.

> **Scope of this stage:** only the **UNet** and **VAE (encoder + decoder)**
> exporters are provided here. The **Qwen3-VL-2B text encoder** is *not* covered
> yet — it needs a separate LLM runtime story (see [Text Encoder](#text-encoder-not-yet)).

## Directory Structure

```
deploy/android/
├── _export_common.py            # Shared: QNN-friendly attention, ONNX export, FP16, parity check
├── export_unet_onnx.py          # Export UNet  -> unet.onnx
├── export_vae_decoder_onnx.py   # Export VAE decoder -> vae_decoder.onnx
├── export_vae_encoder_onnx.py   # Export VAE encoder -> vae_encoder.onnx
├── fuse_rmsnorm_onnx.py         # Fuse RMSNorm arithmetic into ONNX RMSNormalization
├── fuse_groupnorm_onnx.py       # Fuse GroupNorm lowering into ONNX GroupNormalization
└── README.md                    # This file
```

## 1. Requirements

| Component | Requirement |
|-----------|-------------|
| Model export | Python 3.10+, PyTorch 2.3+, `diffusers`, this repo's `dreamlite` package |
| ONNX tooling | `onnx`, `onnxruntime`, `onnxsim`, `onnxconverter-common` |
| QNN compilation | Qualcomm QNN SDK (a.k.a. QAIRT) — run on your own machine |
| Device | Snapdragon 8 Gen 2 / Gen 3 / 8 Elite (Hexagon NPU), 8 GB+ RAM |

Install the ONNX tooling (export runs on CPU, no GPU required):

```bash
pip install onnx onnxruntime onnxsim onnxconverter-common
```

Ensure the weights are laid out as expected (same as the main README):

```
DreamLite/
├── models/
│   └── DreamLite-mobile/
│       ├── unet/
│       └── vae/
```

## 2. Model Export

Run from the **repository root** (the scripts import the local `dreamlite`
package). All artifacts are written to `exported_models/android/`.

```bash
# UNet: FP32 ONNX, simplified, plus an FP16 copy
python deploy/android/export_unet_onnx.py --simplify --fp16

# VAE decoder (always needed) and encoder (edit mode only)
python deploy/android/export_vae_decoder_onnx.py --simplify --fp16
python deploy/android/export_vae_encoder_onnx.py --simplify --fp16
```

For ByteNN, fuse the exported RMSNorm arithmetic into the standard ONNX
`RMSNormalization` operator (opset 23):

```bash
python deploy/android/fuse_rmsnorm_onnx.py \
  exported_models/android/unet_qnn4d.onnx \
  exported_models/android/unet_bytenn.onnx

python -c "from deploy.android._export_common import convert_fp16; from pathlib import Path; convert_fp16(Path('exported_models/android/unet_bytenn.onnx'), Path('exported_models/android/unet_bytenn.fp16.onnx'))"
```

The PyTorch exporter does not reliably emit the standard ONNX
`GroupNormalization` node. It commonly lowers `torch.nn.GroupNorm` to
`Reshape → InstanceNormalization → Reshape → Mul → Add`; the flattened
`InstanceNormalization` input can be much larger than the original feature
map. Fuse that exact pattern as a separate post-pass:

```bash
python deploy/android/fuse_groupnorm_onnx.py \
  exported_models/android/unet_bytenn.onnx \
  exported_models/android/unet_bytenn_gn.onnx

python -c "from deploy.android._export_common import convert_fp16; from pathlib import Path; convert_fp16(Path('exported_models/android/unet_bytenn_gn.onnx'), Path('exported_models/android/unet_bytenn_gn.fp16.onnx'))"
```

`GroupNormalization` is a standard ONNX operator from opset 18. The post-pass
preserves the original GroupNorm parameters and epsilon, and only replaces a
complete single-consumer lowering. It does not replace ordinary
`LayerNormalization` or unrelated `InstanceNormalization` nodes.

The resulting ByteNN artifacts are:

```
exported_models/android/unet_bytenn.onnx          # FP32, opset 23
exported_models/android/unet_bytenn.fp16.onnx     # FP16 weights, opset 23
exported_models/android/unet_bytenn_gn.onnx       # FP32, GroupNormalization fused
exported_models/android/unet_bytenn_gn.fp16.onnx  # FP16, GroupNormalization fused
```

Outputs:

```
exported_models/android/
├── unet.onnx            (+ unet.fp16.onnx)
├── vae_decoder.onnx     (+ vae_decoder.fp16.onnx)
└── vae_encoder.onnx     (+ vae_encoder.fp16.onnx)
```

### Common flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--seq-len N` | `128` | *(UNet only)* Fixed text sequence length baked into the static graph. |
| `--fp16` | off | Also emit an FP16 copy (new HTP runs FP16 natively, no calibration). |
| `--simplify` | off | Run onnx-simplifier to fold/clean the graph. |
| `--no-parity` | off | Skip the PyTorch↔onnxruntime parity check. |
| `--opset N` | `17` | ONNX opset version. |

Each script prints a **parity report** comparing the PyTorch reference output
against onnxruntime (CPU):

```
[parity] max_abs_err=3.1e-04  max_rel_err=8.7e-04  within(rtol=0.01, atol=0.001)=True
```

## 3. Export Design Notes (why the Android graph differs from iOS)

QNN HTP has two hard constraints that shape these exporters:

1. **Static shapes.** Unlike the iOS CoreML export (which used a dynamic
   `RangeDim` for the text sequence length), every dimension here is fixed.
   The UNet's text length is `--seq-len` (default 128). **At runtime, pad the
   text embeddings and attention mask to exactly this length**, setting the mask
   to `0` on the padding — the UNet turns the mask into a `-10000` additive bias
   (see [`../../dreamlite/models/unets/unet_2d_condition_mobile.py`](../../dreamlite/models/unets/unet_2d_condition_mobile.py)),
   so padded positions contribute nothing.

2. **No SDPA / no head-dim broadcast.** The UNet uses grouped-query attention
   (`num_kv_heads=1`) and relies on `F.scaled_dot_product_attention` broadcasting
   1 KV head across all query heads. QNN converters lower this poorly. During
   export we swap in `ExportFriendlyAttnProcessor` (`_export_common.py`), which:
   - materialises K/V to the full head count with a 4D `repeat`, and
   - replaces SDPA with explicit `matmul → scale → (+mask) → softmax → matmul`.

   This produces a static graph with a single head count and no fused SDPA op.

3. **RMSNorm fusion for ByteNN.** Attention q/k normalization and the text
   projection use RMSNorm, not ordinary LayerNorm. The PyTorch RMSNorm export
   is normally `Pow → ReduceMean → Add → Sqrt → Div → Mul`. The
   `fuse_rmsnorm_onnx.py` post-pass replaces only complete matching chains with
   the standard ONNX `RMSNormalization(X, scale)` operator (opset 23,
   `axis=-1`, `epsilon=1e-5`, `stash_type=1`). The 108 ordinary Transformer
   `LayerNormalization` nodes are left unchanged.

Shapes (batch = 1, in-context width = 2× latent width):

| Model | Inputs | Output |
|-------|--------|--------|
| UNet | `sample[1,4,128,256]`, `timestep[1]`, `encoder_hidden_states[1,SEQ,2048]`, `encoder_attention_mask[1,SEQ]`, `time_ids[1,2]` | `noise_pred[1,4,128,256]` |
| VAE decoder | `latent[1,4,128,128]` | `image[1,3,1024,1024]` |
| VAE encoder | `image[1,3,1024,1024]` | `latent[1,4,128,128]` |

## 4. Next Step: QNN Compilation

The scripts stop at `.onnx`. To run on the Hexagon NPU you then compile with the
QNN SDK on a machine that has QAIRT installed. Two common paths:

**A. ONNX Runtime QNN Execution Provider** (recommended — keeps ORT as the API):
build/generate a QNN context binary via ORT's QNN EP, then load it on-device
through `onnxruntime-android` with the QNN EP enabled. This is the smoothest fit
for the "ONNX Runtime" runtime decision.

**B. Native QNN converter** (standalone QNN graphs):

```bash
# FP16 on HTP (no calibration needed)
qnn-onnx-converter --input_network unet.onnx --float_bitwidth 16 --output_path unet.cpp
qnn-model-lib-generator -c unet.cpp -b unet.bin -t x86_64-linux-clang
qnn-context-binary-generator --model libunet.so --backend libQnnHtp.so --binary_file unet
```

> Static input shapes are already baked in, so no `--input_dim` overrides are
> needed. For INT8/INT16 quantization (a later optimization) you would supply a
> calibration `--input_list`; this stage ships FP32/FP16 only.

## 5. Runtime Data Flow (to port to Kotlin/C++)

Identical to iOS — reuse the Swift files as reference:

```
prompt (+ optional source image)
  → Text Encoder (Qwen3-VL): template → tokenize → last hidden states
      → drop leading tokens (generate=34 / edit=64) → pad/trim to SEQ → [1,SEQ,2048] + mask
  → init random latent [1,4,128,128]
  → cond latent: generate = zeros / edit = vae_encoder(source image)
  → denoise loop (4 steps):
        concat width → sample [1,4,128,256]
        unet.onnx(sample, timestep, hidden_states, mask, time_ids) → [1,4,128,256]
        crop first half → [1,4,128,128]
        scheduler.step: latent += (σ_next − σ) · noise_pred
  → vae_decoder.onnx(latent) → image [1,3,1024,1024]   (scaling=1, shift=0, no rescale)
```

The scheduler math (dynamic-shift `mu`, exponential time-shift, Euler step) is in
[`../FluxScheduler.swift`](../FluxScheduler.swift); prompt templates, drop
indices, and image preprocessing are in [`../MLXTextEncoder.swift`](../MLXTextEncoder.swift).

## Text Encoder (not yet)

The Qwen3-VL-2B text encoder is intentionally **out of scope** for this export
stage. It is not a good ONNX/QNN fit as-is:

- **generate** mode needs only the text tower's last hidden states (feasible as a
  fixed-seq ONNX graph later);
- **edit** mode additionally runs the vision tower + deepstack + interleaved
  M-RoPE, which is heavy and QNN-unfriendly.

The likely Android approach is a dedicated LLM runtime (e.g. MNN-LLM or
llama.cpp) exposing last-hidden-states, analogous to how iOS uses MLX. This will
be handled in a separate stage.

## Notes

- FP32 is the primary artifact (numerically checkable); FP16 is provided for
  direct HTP execution. INT8/INT16 quantization is a later optimization.
- The exported UNet and VAE cover the full image pipeline; only the text encoder
  remains before an end-to-end Android build is possible.
