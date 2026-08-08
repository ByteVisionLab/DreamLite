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

"""Export the DreamLite-mobile VAE decoder (TAESD-XL) to ONNX.

Mirrors ``deploy/export_vae_decoder.py`` but targets onnxruntime + QNN.
TAESD-XL uses ``scaling_factor=1`` and ``shift_factor=0``, so no latent
rescale is applied here (the pipeline feeds latents directly).

I/O (static, batch=1):
    latent  [1, 4, 128, 128]  ->  image [1, 3, 1024, 1024]   (range [0, 1])
"""

import argparse
from pathlib import Path

import torch

from _export_common import (
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
OUTPUT_NAME = "vae_decoder.onnx"
# =============================


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


def parse_args():
    p = argparse.ArgumentParser(description="Export DreamLite-mobile VAE decoder to ONNX (QNN-friendly).")
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default=OUTPUT_NAME)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true", help="Also emit an FP16 copy (vae_decoder.fp16.onnx).")
    p.add_argument("--simplify", action="store_true", help="Run onnx-simplifier on the graph.")
    p.add_argument("--no-parity", action="store_true", help="Skip the PyTorch<->ORT parity check.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_path = output_dir / args.output_name

    # 1. Load VAE.
    print("Loading VAE...")
    from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

    vae = AutoencoderTiny.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")

    wrapper = VAEDecoderWrapper(vae).eval()

    # 2. Dummy input: 1024x1024 -> 128x128 latent (8x downsample).
    dummy_latent = torch.randn(1, 4, 128, 128)
    input_names = ["latent"]
    output_names = ["image"]
    print(f"  Input shape: {tuple(dummy_latent.shape)}")

    # 3. PyTorch reference.
    print("\nTesting PyTorch inference...")
    with torch.no_grad():
        torch_out = wrapper(dummy_latent)
    print(f"  Output shape: {tuple(torch_out.shape)}")  # [1, 3, 1024, 1024]

    # 4. Export.
    print("\nExporting to ONNX...")
    export_onnx(wrapper, (dummy_latent,), input_names, output_names, save_path, opset=args.opset)

    # 5. Post-process.
    if args.simplify:
        simplify_onnx(save_path)
    check_onnx(save_path)
    report_size(save_path)

    # 6. Parity.
    if not args.no_parity:
        feeds = to_numpy_feeds(input_names, (dummy_latent,))
        check_parity(save_path, torch_out, feeds, output_names[0])

    # 7. Optional FP16.
    if args.fp16:
        fp16_path = save_path.with_suffix(".fp16.onnx")
        convert_fp16(save_path, fp16_path)
        report_size(fp16_path)

    print(f"\n✅ VAE decoder exported: {save_path}")


if __name__ == "__main__":
    main()
