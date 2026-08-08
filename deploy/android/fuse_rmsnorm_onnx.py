# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fuse exported RMSNorm arithmetic subgraphs into ONNX RMSNormalization.

The DreamLite UNet uses RMSNorm for attention q/k normalization and for the
text projection.  PyTorch exports this as Pow -> ReduceMean -> Add -> Sqrt ->
Div -> Mul -> Mul.  ONNX RMSNormalization was introduced in opset 23 and
expresses the same operation as RMSNormalization(X, scale).

This script is deliberately conservative: it only removes a complete,
single-consumer RMSNorm chain whose exponent is 2, reduction axis is -1,
epsilon is a scalar constant, reciprocal numerator is 1, and final scale is
an initializer.  Ordinary LayerNormalization nodes are left untouched.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


RMS_DOMAIN = ""
RMS_OPSET = 23


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


def _scalar_equal(values: Dict[str, np.ndarray], name: str, expected: float, atol: float = 1e-6) -> bool:
    value = values.get(name)
    return value is not None and value.size == 1 and bool(np.isclose(float(value.reshape(-1)[0]), expected, atol=atol))


def _axes_are_last(values: Dict[str, np.ndarray], name: str) -> bool:
    value = values.get(name)
    return value is not None and value.size == 1 and int(value.reshape(-1)[0]) == -1


def _single_producer(producers: Dict[str, onnx.NodeProto], name: str, op_type: str) -> Optional[onnx.NodeProto]:
    node = producers.get(name)
    return node if node is not None and node.op_type == op_type else None


def _match_rmsnorm(
    final: onnx.NodeProto,
    producers: Dict[str, onnx.NodeProto],
    consumers: Dict[str, List[onnx.NodeProto]],
    values: Dict[str, np.ndarray],
) -> Optional[Tuple[onnx.NodeProto, List[onnx.NodeProto], str, float]]:
    if final.op_type != "Mul" or len(final.input) != 2:
        return None

    # The final Mul is x / rms * scale.  The scale must be a model initializer
    # (not an arbitrary runtime tensor), and the other input must be the
    # normalized tensor.
    scale_name = next((x for x in final.input if x in values and values[x].ndim == 1), None)
    normalized_name = next((x for x in final.input if x != scale_name), None)
    if scale_name is None or normalized_name is None:
        return None

    mul_normalized = _single_producer(producers, normalized_name, "Mul")
    if mul_normalized is None or len(consumers.get(normalized_name, [])) != 1:
        return None

    x_name = next((x for x in mul_normalized.input if x != normalized_name), None)
    inv_name = next((x for x in mul_normalized.input if x != x_name), None)
    if x_name is None or inv_name is None:
        return None

    div = _single_producer(producers, inv_name, "Div")
    if div is None or len(consumers.get(inv_name, [])) != 1:
        return None
    numerator = next((x for x in div.input if _scalar_equal(values, x, 1.0)), None)
    denominator = next((x for x in div.input if x != numerator), None)
    if numerator is None or denominator is None:
        return None

    sqrt = _single_producer(producers, denominator, "Sqrt")
    if sqrt is None or len(consumers.get(denominator, [])) != 1:
        return None
    add = _single_producer(producers, sqrt.input[0], "Add")
    if add is None or len(consumers.get(sqrt.input[0], [])) != 1:
        return None

    epsilon_name = next((x for x in add.input if x in values and values[x].size == 1), None)
    mean_name = next((x for x in add.input if x != epsilon_name), None)
    if epsilon_name is None or mean_name is None:
        return None
    epsilon = float(values[epsilon_name].reshape(-1)[0])

    reduce_mean = _single_producer(producers, mean_name, "ReduceMean")
    if reduce_mean is None or len(consumers.get(mean_name, [])) != 1 or len(reduce_mean.input) != 1:
        return None
    axes_name = next((x for x in reduce_mean.input[1:] if _axes_are_last(values, x)), None)
    if axes_name is None:
        # Opset 17 exports axes as an attribute; support that form too.
        axes = next((a for a in reduce_mean.attribute if a.name == "axes"), None)
        if axes is None or list(axes.ints) != [-1]:
            return None
    squared_name = reduce_mean.input[0]

    power = _single_producer(producers, squared_name, "Pow")
    if power is None or len(consumers.get(squared_name, [])) != 1 or len(power.input) != 2:
        return None
    if not _scalar_equal(values, power.input[1], 2.0):
        return None

    if power.input[0] != x_name:
        return None

    chain = [power, reduce_mean, add, sqrt, div, mul_normalized, final]
    if any(len(consumers.get(node.output[0], [])) != 1 for node in chain[:-1]):
        return None
    return final, chain, x_name, epsilon


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
        match = _match_rmsnorm(node, producers, consumers, values)
        if match is None:
            continue
        final, chain, x_name, epsilon = match
        scale_name = next(x for x in final.input if x in values and values[x].ndim == 1)
        replacement = helper.make_node(
            "RMSNormalization",
            inputs=[x_name, scale_name],
            outputs=list(final.output),
            name=final.name.rsplit("/Mul", 1)[0] + "/RMSNormalization",
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
        )
        replacements[index] = replacement
        removed.update(id(item) for item in chain)
        fused += 1

    new_nodes: List[onnx.NodeProto] = []
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
            opset.version = max(opset.version, RMS_OPSET)
            break
    else:
        model.opset_import.append(helper.make_opsetid("", RMS_OPSET))
    return model, fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse RMSNorm arithmetic into ONNX RMSNormalization.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.input), load_external_data=False)
    model, fused = fuse_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"Fused RMSNormalization nodes: {fused}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
