import tensorrt as trt
from datetime import datetime
import json
import time
import os
import numpy as np
from glob import glob
import cv2

if __package__:
    from .calibrator import Calibrator
else:
    from calibrator import Calibrator


def preprocess(img):
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


class DataLoader:

    def __init__(self, image_dir: str, batch_size: int = 1, limit: int = 50):
        self.index = 0
        self.length = limit
        self.batch_size = batch_size
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(image_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, (
            "{} must contains more than ".format(limit)
            + str(self.batch_size * self.length)
            + " images to calib"
        )
        print("found all {} images to calib.".format(len(self.img_list)))
        self.calibration_data = np.zeros(
            (self.batch_size, 3, height, width), dtype=np.float32
        )

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(
                    self.img_list[i + self.index * self.batch_size]
                ), "not found!!"
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


if __name__ == "__main__":

    height = 640
    width = 640
    ONE_GB_IN_BYTES = 1024 * 1024 * 1024
    prefix = "TensorRT:"
    f_onnx = r"models/det-5n-int8.onnx"
    image_dir = "data/calib"

    # region common logic
    print("Input:", f_onnx)
    print("Output:", f_onnx)
    workspace_in_gb = 3
    workspace_in_bytes = workspace_in_gb * ONE_GB_IN_BYTES
    print("workspace_in_gb    :", workspace_in_gb)
    print("workspace_in_bytes :", workspace_in_bytes)

    fname = os.path.basename(f_onnx)
    fname_wo_ext = fname[: fname.rindex(".")]
    f_output = f"models/{fname_wo_ext}-int8.engine"
    if os.path.exists(f_output):
        print(f"engine file already exists at: {f_output}")
        exit(0)
    print("Generating engine file at:", f_output)
    # endregion

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

    # use int8
    config.set_flag(trt.BuilderFlag.INT8)
    data_loader = DataLoader(image_dir, batch_size=1, limit=50)
    calibrator = Calibrator(stream=data_loader, cache_file="calib.cache")
    config.set_int8_calibrator(calibrator)
    print("writing engine file")
    # Write file
    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine:
        with open(f_output, "wb") as t:
            t.write(bytearray(engine.serialize()))

    elapsed_time = round((time.time() - start_time) / 60, 1)
    print(f"conversion completed in {elapsed_time} mins")
