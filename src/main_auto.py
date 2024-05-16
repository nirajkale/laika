import os
import Jetson.GPIO as GPIO
from adafruit_servokit import ServoKit
import time
from utils.display_utils import DisplayManager
from utils.gst_utils import get_video_capture, get_video_writer
from utils.logger import Logger, LogLevel
from utils.hardware_utils import Position
from utils.vision_utils import draw_text_on_image
from rich.console import Console, Text
from detectron import BaseDetector, TensorRTDetector
import gc
import cv2
import traceback
from typing import Callable, List, Tuple
from utils.vision_utils import Detection

os.environ["LOG_LEVEL"] = "DEBUG"

def get_unified_logger(logger: Logger, disp: DisplayManager)-> Callable[[LogLevel, str, str], None]:
    def logging_fn(log_level: LogLevel, short_text:str, long_text: str):
        disp.print_line(text=short_text, clear=True, line_num=0)
        logger.log(log_level=log_level, message=long_text)
    return logging_fn

def process_detections(detections: List[Detection], last_pos: Position, beta1: float=0.2)-> Position:
    face, human = None, None
    for detection in detections:
        if detection.label == "face":
            face = detection
        elif detection.label == "human":
            human = detection
        if face is not None and human is None:
            break
    obj = face if face else human
    if not obj:
        return last_pos
    obj.normalize_dims()
    pitch_target = 
    # servo_h = int(abs(obj.center_x - 0.5) * 180 * beta1)
    # servo_v = int(abs(obj.center_y - 0.5) * 130 * beta1)
    servo_h = int(obj.center_x * 180)
    servo_v = int(obj.center_y * 130)
    return Position(x=obj.center_x, y=obj.center_y, servo_h=servo_h, servo_v=servo_v)
    

def main(*args, **kwargs):
    # define variables that need to disposed off
    servo_kit = None
    disp, console = None, None
    cap, out = None, None
    position = Position(x=0.5, y=0.5, servo_h=90, servo_v=45)
    try:
        #region display: 16x2 lcd + console + logger init
        disp = DisplayManager(line_height=12)
        console = Console(emoji=True)
        log_level = LogLevel.from_str(kwargs.get("log_level", os.environ.get("LOG_LEVEL", "INFO")))
        logger = Logger("Main", log_level=log_level, console=console)
        update = get_unified_logger(logger, disp)
        #endregion
        #region hardware pins init
        servo_kit = ServoKit(channels=16)
        # servo_kit.servo[1].angle = 0
        servo_kit.servo[0].angle = position.servo_h
        servo_kit.servo[1].angle = position.servo_v
        # mux setup
        update(LogLevel.INFO, "hardware init", "initializing hardware")
        pwm_pin1, pwm_pin2 = "GPIO_PE6", "LCD_BL_PW"
        s0_pin, s1_pin, s2_pin = "SPI2_CS1", "SPI2_CS0", "SPI2_MISO"
        GPIO.setup([pwm_pin1, pwm_pin2, s0_pin, s1_pin, s2_pin], GPIO.OUT)
        pi_pwm1 = GPIO.PWM(pwm_pin1, 100)
        pi_pwm2 = GPIO.PWM(pwm_pin2, 100)
        pi_pwm1.start(0)
        pi_pwm2.start(0)
        #endregion
        #region gstreamer cap & out init
        update(LogLevel.INFO, "gstreamer init", "initializing gstreamer pipelines")
        cap = get_video_capture(**kwargs)
        out = get_video_writer(**kwargs)
        if not cap.isOpened():
            raise Exception("Video capture not available")
        if not out.isOpened():
            raise Exception("Video writer not available")
        #endregion
        #region model init & warmup
        update(LogLevel.INFO, "model init...", "initializing obj detection model")
        detector = TensorRTDetector(
            model_path=kwargs.pop("model_path"), 
            classes=kwargs.pop("object_det_classes"), 
            device=kwargs.pop("device"), 
            console=console,
            **kwargs
        )
        #endregion
        #region event loop
        update(LogLevel.INFO, "starting loop...", "starting event loop")
        prev_frame_time = 0
        new_frame_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: VideoCapture failed to read frame.")
                break
            new_frame_time = time.time()
            fps = round(1 / (new_frame_time - prev_frame_time), 2)
            # iteration_delay_ms = int((new_frame_time - prev_frame_time) * 1000)
            prev_frame_time = new_frame_time
            img_out, detections, latency_info = detector.predict(source=frame, isBGR=True)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            position = process_detections(detections, last_pos=position)
            img_out = draw_text_on_image(img_out, \
                                        text=f"FPS: {fps:.1f} | x:{position.servo_h}, y:{position.servo_v}", \
                                        position=(10, 50))
            img_out = draw_text_on_image(img_out, \
                                        text=screen_augmentation_str, \
                                        position=(10, 50))
            out.write(img_out)
            # disp.print_line(f"FPS: {fps:.2f}", clear=False, line_num=1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        #endregion
    except KeyboardInterrupt:
        console.print("Event-loop break requested :cross_mark:")
    except Exception as e:
        console.print(e, style="bold red")
        traceback.print_exc()
    finally:
        if cap and cap.isOpened():
            cap.release()
        if out and out.isOpened():
            out.release()
        GPIO.cleanup()
        if disp:
            disp.clear()
        del console
        gc.collect()

if __name__ == "__main__":

    config = {
        "width" : 640,
        "height": 640,
        "frame_rate": 10,
        "host_ip_addr": "192.168.3.2",
        "model_path": "models/det-5n-half.engine",
        "object_det_classes": ["face", "human"],
        "device": 0,
        "warmup": True,
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }

    main(**config)
    print('done!')

"""
servo_kit.servo[0], horizontal motion with 90 as mid point
servo_kit.servo[1], vertical motion with 75 as mid point
"""