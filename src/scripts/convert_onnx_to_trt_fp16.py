import tensorrt as trt
from datetime import datetime
import json
import time
import os
import numpy as np


ONE_GB_IN_BYTES = 1024 * 1024 * 1024
prefix = "TensorRT:"

f_onnx = r"models/det-5n.onnx"
print("Input:", f_onnx)
print("Output:", f_onnx)
workspace_in_gb = 3
workspace_in_bytes = workspace_in_gb * ONE_GB_IN_BYTES
print("workspace_in_gb    :", workspace_in_gb)
print("workspace_in_bytes :", workspace_in_bytes)
metadata = {
    "description": "Yolov8n model as detectron",
    "author": "Niraj Kale",
    "date": datetime.now().isoformat(),
    "version": "1.0",
    "batch": 1,
    "imgsz": 640,
}

fname = os.path.basename(f_onnx)
fname_wo_ext = fname[: fname.rindex(".")]
f_output = f"models/{fname_wo_ext}-half.engine"
print("Generating engine file at:", f_output)
if os.path.exists(f_output):
    os.remove(f_output)

start_time = time.time()
is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
print("is_trt10:", is_trt10)

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
config = builder.create_builder_config()
if is_trt10:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_in_bytes)
else:  # TensorRT versions 7, 8
    config.max_workspace_size = workspace_in_bytes

flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
print("parsing onnx file")
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(f_onnx):
    raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]
for inp in inputs:
    print(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
for out in outputs:
    print(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

if builder.platform_has_fast_fp16:
    print("converting to FP16")
    config.set_flag(trt.BuilderFlag.FP16)
else:
    raise NotImplementedError()
print("writing engine file")
# Write file
build = builder.build_serialized_network if is_trt10 else builder.build_engine
with build(network, config) as engine:
    with open(f_output, "wb") as t:
        t.write(bytearray(engine.serialize()))

elapsed_time = round((time.time() - start_time) / 60, 1)
print(f"conversion completed in {elapsed_time} mins")
