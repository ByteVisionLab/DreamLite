# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fuse PyTorch's GroupNorm ONNX lowering into GroupNormalization.

The PyTorch exporter commonly lowers ``GroupNorm(C, G)`` to:

    Reshape(N, C, H, W) -> [N*G, C/G, H*W]
        -> InstanceNormalization -> Reshape(N, C, H, W)

The standard ONNX GroupNormalization operator represents this operation
directly and avoids the large flattened intermediate tensor.  This pass only
matches the complete, single-consumer pattern with static shape constants,
matching input/output element counts, and per-channel affine parameters.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper


GROUPNORM_OPSET = 18


def _constant_values(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        values[initializer.name] = numpy_helper.to_array(initializer)
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                values[node.output[0]] = numpy_helper.to_array(attribute.t)
                break
    return values


def _shape(values: Dict[str, np.ndarray], name: str) -> Optional[Tuple[int, ...]]:
    value = values.get(name)
    if value is None or value.ndim != 1:
        return None
    return tuple(int(item) for item in value.tolist())


def _reshape_pattern(values: Dict[str, np.ndarray], name: str) -> Optional[Tuple[int, ...]]:
    shape = _shape(values, name)
    if shape is None or len(shape) == 0:
        return None
    if any(item < -1 for item in shape) or shape.count(-1) > 1:
        return None
    return shape


def _single_producer(
    producers: Dict[str, onnx.NodeProto], name: str, op_type: str
) -> Optional[onnx.NodeProto]:
    node = producers.get(name)
    return node if node is not None and node.op_type == op_type else None


def _epsilon(node: onnx.NodeProto) -> Optional[float]:
    for attribute in node.attribute:
        if attribute.name == "epsilon":
            return float(attribute.f)
    return None


def _match(
    instance_norm: onnx.NodeProto,
    producers: Dict[str, onnx.NodeProto],
    consumers: Dict[str, List[onnx.NodeProto]],
    values: Dict[str, np.ndarray],
) -> Optional[Tuple[List[onnx.NodeProto], str, str, str, int, float]]:
    if instance_norm.op_type != "InstanceNormalization" or len(instance_norm.input) != 3:
        return None
    if len(instance_norm.output) != 1:
        return None

    pre_reshape = _single_producer(producers, instance_norm.input[0], "Reshape")
    if pre_reshape is None or len(pre_reshape.input) != 2:
        return None
    if len(consumers.get(pre_reshape.output[0], [])) != 1:
        return None

    post_reshape = None
    output_name = instance_norm.output[0]
    if len(consumers.get(output_name, [])) == 1:
        candidate = consumers[output_name][0]
        if candidate.op_type == "Reshape" and len(candidate.input) == 2:
            post_reshape = candidate
    if post_reshape is None or len(consumers.get(post_reshape.output[0], [])) > 1:
        return None

    input_name = pre_reshape.input[0]
    input_pattern = _reshape_pattern(values, pre_reshape.input[1])
    output_shape = _shape(values, post_reshape.input[1])
    if input_pattern is None or output_shape is None:
        return None
    if len(input_pattern) != 3:
        return None
    if input_pattern[0] != 0 or input_pattern[1] <= 1 or input_pattern[2] != -1:
        return None
    if len(output_shape) < 3 or any(item <= 0 for item in output_shape):
        return None

    affine_nodes: List[onnx.NodeProto] = []
    affine_output = post_reshape.output[0]
    affine_scale_name: Optional[str] = None
    affine_bias_name: Optional[str] = None
    next_nodes = consumers.get(affine_output, [])
    if len(next_nodes) == 1 and next_nodes[0].op_type == "Mul":
        affine_mul = next_nodes[0]
        affine_scale_name = next(
            (name for name in affine_mul.input if name in values), None
        )
        if affine_scale_name is not None:
            affine_output = affine_mul.output[0]
            affine_nodes.append(affine_mul)
            next_nodes = consumers.get(affine_output, [])
    if len(next_nodes) == 1 and next_nodes[0].op_type == "Add":
        affine_add = next_nodes[0]
        affine_bias_name = next(
            (name for name in affine_add.input if name in values), None
        )
        if affine_bias_name is not None:
            affine_output = affine_add.output[0]
            affine_nodes.append(affine_add)

    if affine_scale_name is None or affine_bias_name is None:
        scale_name, bias_name = instance_norm.input[1:]
    else:
        scale_name, bias_name = affine_scale_name, affine_bias_name

    scale = values.get(scale_name)
    bias = values.get(bias_name)
    if scale is None or bias is None:
        return None
    if affine_nodes:
        if scale.ndim != 3 or bias.ndim != 3:
            return None
        scale = scale.reshape(-1)
        bias = bias.reshape(-1)
    if scale.ndim != 1 or bias.ndim != 1:
        return None
    if scale.size != bias.size or scale.size == 0:
        return None
    channel_count = scale.size
    if output_shape[1] != channel_count:
        return None
    num_groups = input_pattern[1]
    if num_groups <= 1 or channel_count % num_groups != 0:
        return None

    epsilon = _epsilon(instance_norm)
    if epsilon is None:
        return None

    chain = [pre_reshape, instance_norm, post_reshape] + affine_nodes
    return (
        chain,
        input_name,
        scale_name,
        bias_name,
        num_groups,
        epsilon,
    )


def fuse_model(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, int]:
    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    values = _constant_values(model)

    replacements: Dict[int, onnx.NodeProto] = {}
    removed = set()
    fused = 0
    for index, node in enumerate(nodes):
        match = _match(node, producers, consumers, values)
        if match is None:
            continue
        chain, input_name, scale_name, bias_name, num_groups, epsilon = match
        replacement = helper.make_node(
            "GroupNormalization",
            inputs=[input_name, scale_name, bias_name],
            outputs=list(chain[-1].output),
            name=chain[1].name.rsplit("/InstanceNormalization", 1)[0]
            + "/GroupNormalization",
            num_groups=num_groups,
            epsilon=epsilon,
            stash_type=1,
        )
        replacements[index] = replacement
        removed.update(id(item) for item in chain)
        fused += 1

    new_nodes = []
    for index, node in enumerate(nodes):
        if id(node) in removed:
            if index in replacements:
                new_nodes.append(replacements[index])
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, GROUPNORM_OPSET)
            break
    else:
        model.opset_import.append(helper.make_opsetid("", GROUPNORM_OPSET))
    return model, fused


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse GroupNorm lowering into ONNX GroupNormalization."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.input), load_external_data=False)
    model, fused = fuse_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"Fused GroupNormalization nodes: {fused}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
