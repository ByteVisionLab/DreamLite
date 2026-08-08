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

"""Export the DreamLite-mobile VAE encoder (TAESD-XL) to ONNX.

Mirrors ``deploy/export_vae_encoder.py`` but targets onnxruntime + QNN.
Used only in **edit** mode to turn the source image into a condition latent.

I/O (static, batch=1):
    image   [1, 3, 1024, 1024]  (normalized to [-1, 1])  ->  latent [1, 4, 128, 128]
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
OUTPUT_NAME = "vae_encoder.onnx"
# =============================


class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        return self.vae.encode(image, return_dict=False)[0]


def parse_args():
    p = argparse.ArgumentParser(description="Export DreamLite-mobile VAE encoder to ONNX (QNN-friendly).")
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--output-name", type=str, default=OUTPUT_NAME)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fp16", action="store_true", help="Also emit an FP16 copy (vae_encoder.fp16.onnx).")
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

    wrapper = VAEEncoderWrapper(vae).eval()

    # 2. Dummy input: image in [-1, 1].
    dummy_image = torch.randn(1, 3, 1024, 1024).clamp(-1, 1)
    input_names = ["image"]
    output_names = ["latent"]
    print(f"  Input shape: {tuple(dummy_image.shape)}")

    # 3. PyTorch reference.
    print("\nTesting PyTorch inference...")
    with torch.no_grad():
        torch_out = wrapper(dummy_image)
    print(f"  Output shape: {tuple(torch_out.shape)}")  # [1, 4, 128, 128]

    # 4. Export.
    print("\nExporting to ONNX...")
    export_onnx(wrapper, (dummy_image,), input_names, output_names, save_path, opset=args.opset)

    # 5. Post-process.
    if args.simplify:
        simplify_onnx(save_path)
    check_onnx(save_path)
    report_size(save_path)

    # 6. Parity.
    if not args.no_parity:
        feeds = to_numpy_feeds(input_names, (dummy_image,))
        check_parity(save_path, torch_out, feeds, output_names[0])

    # 7. Optional FP16.
    if args.fp16:
        fp16_path = save_path.with_suffix(".fp16.onnx")
        convert_fp16(save_path, fp16_path)
        report_size(fp16_path)

    print(f"\n✅ VAE encoder exported: {save_path}")


if __name__ == "__main__":
    main()
