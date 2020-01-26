import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

while True:
    kit.servo[0].angle = 45
    time.sleep(1)
    kit.servo[1].angle = 120
    time.sleep(2)

    kit.servo[0].angle = 150
    time.sleep(1)
    kit.servo[1].angle = 160
    time.sleep(2)

