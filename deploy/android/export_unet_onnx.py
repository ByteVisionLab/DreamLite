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

"""Export the DreamLite-mobile UNet to a QNN-friendly ONNX graph.

Mirrors ``deploy/export_unet.py`` (the CoreML/iOS exporter) but targets
onnxruntime + Qualcomm QNN. Two deliberate differences from the iOS path:

* **Static shapes** — QNN HTP requires fixed dimensions, so the text sequence
  length is a fixed ``--seq-len`` (default 128) instead of a CoreML RangeDim.
  At runtime, pad the text embeddings/mask to this length and zero the mask.
* **Export-friendly attention** — the grouped-query (``kv_heads=1``) SDPA is
  rewritten into explicit matmul/softmax with the K/V materialised to the full
  head count (see ``_export_common.ExportFriendlyAttnProcessor``).

I/O (all batch=1, in-context width = 2 × latent width):
    sample                  [1, 4, 128, 256]   noisy || cond latent (width-concat)
    timestep                [1]
    encoder_hidden_states   [1, SEQ, 2048]     Qwen3-VL last hidden states
    encoder_attention_mask  [1, SEQ]
    time_ids                [1, 2]             (width, height)
  ->
    noise_pred              [1, 4, 128, 256]
"""

import argparse
from pathlib import Path

import torch

from _export_common import (
    apply_export_friendly_attention,
    check_onnx,
    check_parity,
    convert_fp16,
    export_onnx,
    report_size,
    simplify_onnx,
    to_numpy_feeds,
)

# ========= Config ============
MODEL_PATH = "models/DreamLite-mobile"
OUTPUT_DIR = Path("./exported_models/android")
OUTPUT_NAME = "unet.onnx"
# =============================


class UNetWrapper(torch.nn.Module):
    """Flatten the ``added_cond_kwargs`` dict into explicit tensor args (same as iOS)."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, encoder_attention_mask, time_ids):
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )[0]


def parse_args():
    p = argparse.ArgumentParser(description="Export DreamLite-mobile UNet to ONNX (QNN-friendly).")
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default=OUTPUT_NAME)
    p.add_argument("--seq-len", type=int, default=128,
                   help="Fixed text sequence length baked into the static graph.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true", help="Also emit an FP16 copy (unet.fp16.onnx).")
    p.add_argument("--simplify", action="store_true", help="Run onnx-simplifier on the graph.")
    p.add_argument("--no-parity", action="store_true", help="Skip the PyTorch<->ORT parity check.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_path = output_dir / args.output_name

    # 1. Load UNet (FP32 for a clean, checkable export).
    #    Load the module directly (subfolder="unet") rather than via the full
    #    pipeline, so we don't drag in the ~4GB Qwen3-VL text encoder.
    print("Loading UNet...")
    from dreamlite import DreamLiteUNetModel

    unet = DreamLiteUNetModel.from_pretrained(args.model_path, subfolder="unet", torch_dtype=torch.float32)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False
    print(f"  UNet loaded: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params")

    # 2. Swap in the QNN-friendly attention processor (GQA expansion + explicit SDPA).
    apply_export_friendly_attention(unet)
    print("  Applied export-friendly attention (GQA expand + explicit matmul/softmax).")

    wrapper = UNetWrapper(unet).eval()

    # 3. Dummy inputs (static shapes).
    seq = args.seq_len
    dummy_inputs = (
        torch.randn(1, 4, 128, 256),           # sample (in-context, width-concat)
        torch.tensor([500.0]),                 # timestep
        torch.randn(1, seq, 2048),             # encoder_hidden_states
        torch.ones(1, seq),                    # encoder_attention_mask
        torch.tensor([[1024.0, 1024.0]]),      # time_ids (width, height)
    )
    input_names = ["sample", "timestep", "encoder_hidden_states", "encoder_attention_mask", "time_ids"]
    output_names = ["noise_pred"]

    print("  Input shapes:")
    for name, t in zip(input_names, dummy_inputs):
        print(f"    {name:24s}{tuple(t.shape)}")

    # 4. PyTorch reference.
    print("\nTesting PyTorch inference...")
    with torch.no_grad():
        torch_out = wrapper(*dummy_inputs)
    print(f"  Output shape: {tuple(torch_out.shape)}")  # [1, 4, 128, 256]

    # 5. Export.
    print("\nExporting to ONNX...")
    export_onnx(wrapper, dummy_inputs, input_names, output_names, save_path, opset=args.opset)

    # 6. Post-process.
    if args.simplify:
        simplify_onnx(save_path)
    check_onnx(save_path)
    report_size(save_path)

    # 7. Parity check vs onnxruntime.
    if not args.no_parity:
        feeds = to_numpy_feeds(input_names, dummy_inputs)
        check_parity(save_path, torch_out, feeds, output_names[0])

    # 8. Optional FP16.
    if args.fp16:
        fp16_path = save_path.with_suffix(".fp16.onnx")
        convert_fp16(save_path, fp16_path)
        report_size(fp16_path)

    print(f"\n✅ UNet exported: {save_path}")


if __name__ == "__main__":
    main()
