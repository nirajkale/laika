from adafruit_servokit import ServoKit

servo_kit = ServoKit(channels=16)
servo_kit.servo[1].angle = 0


text = input('>>').strip()
while text != "exit":
    v, h = text.split(" ")
    servo_kit.servo[0].angle = int(h)
    servo_kit.servo[1].angle = int(v)
    text = input('>>').strip()