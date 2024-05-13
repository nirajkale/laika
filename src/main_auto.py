import os
import Jetson.GPIO as GPIO
from adafruit_servokit import ServoKit
import time

def configure_pwm_pair(servo_kit: ServoKit, pin_on: int, pin_off, pwm_amgle):
    servo_kit.servo[pin_on].angle = pwm_amgle
    servo_kit.servo[pin_off].angle = 0


def to_bin(n):
    return [
        GPIO.HIGH if ch == "1" else GPIO.LOW for ch in bin(n).replace("0b", "").zfill(3)
    ]


if __name__ == "__main__":

    servo_kit = ServoKit(channels=16)
    servo_kit.servo[1].angle = 0
    # mux setup
    pwm_pin1, pwm_pin2 = "GPIO_PE6", "LCD_BL_PW"
    s0_pin, s1_pin, s2_pin = "SPI2_CS1", "SPI2_CS0", "SPI2_MISO"
    GPIO.setup([pwm_pin1, pwm_pin2, s0_pin, s1_pin, s2_pin], GPIO.OUT)
    pi_pwm1 = GPIO.PWM(pwm_pin1, 100)
    pi_pwm2 = GPIO.PWM(pwm_pin2, 100)
    pi_pwm1.start(0)
    pi_pwm2.start(0)

    

    GPIO.cleanup()
    print('done!')

"""
servo_kit.servo[0], horizontal motion with 90 as mid point
servo_kit.servo[1], vertical motion with 75 as mid point
"""