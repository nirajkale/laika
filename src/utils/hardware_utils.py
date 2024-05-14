from adafruit_servokit import ServoKit
import Jetson.GPIO as GPIO
from dataclasses import dataclass

@dataclass
class Position:

    x: float
    y: float
    servo_v: int
    servo_h: int

def configure_pwm_pair(servo_kit: ServoKit, pin_on: int, pin_off:int, pwm_amgle:int):
    servo_kit.servo[pin_on].angle = pwm_amgle
    servo_kit.servo[pin_off].angle = 0

def to_bin(n:int):
    return [
        GPIO.HIGH if ch == "1" else GPIO.LOW for ch in bin(n).replace("0b", "").zfill(3)
    ]