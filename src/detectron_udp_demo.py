"""Uses src/detectron to detect objects in a webcam feed."""

import cv2
import numpy as np
from utils.gst_utils import *
import time

if __package__:
    from .detectron import BaseDetector, ONNXDetector
else:
    from detectron import BaseDetector, ONNXDetector


if __name__ == "__main__":

    WIDTH = 640
    HEIGHT = 640
    FRAME_RATE = 10
    HOST_IP_ADDR = "192.168.3.2"

    reader_pipeline_str = reader_pipeline(
        flip_method=0,
        capture_width=WIDTH,
        capture_height=HEIGHT,
        display_width=WIDTH,
        display_height=HEIGHT,
        framerate=FRAME_RATE,
    )
    writer_pipeline_str = writer_pipeline(
        host_ip_addr=HOST_IP_ADDR,
        width=WIDTH,
        height=HEIGHT,
        port="5004",
        framerate=FRAME_RATE,
    )
    out = cv2.VideoWriter(
        writer_pipeline_str,
        cv2.CAP_GSTREAMER,
        0,
        float(FRAME_RATE),
        (WIDTH, HEIGHT),
        True,
    )
    cap = cv2.VideoCapture(reader_pipeline_str, cv2.CAP_GSTREAMER)
    prev_frame_time = 0
    new_frame_time = 0
    if not cap.isOpened():
        print("Error: VideoCapture failed to open.")
        exit(-1)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: VideoCapture failed to read frame.")
                break
            new_frame_time = time.time()
            fps = round(1 / (new_frame_time - prev_frame_time), 2)
            prev_frame_time = new_frame_time
            frame = cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            out.write(frame)
            # cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cap.isOpened():
            cap.release()