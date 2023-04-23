#! /usr/bin/env python

import RPi.GPIO as GPIO
import time
import math

class Encoder:
        def __init__(self):
                # Encoder GPIO Pins
                self.ENCODER1CHANNELA = 9
                # self.ENCODER1CHANNELB = 11
                self.ENCODER2CHANNELA = 25
                # self.ENCODER2CHANNELB = 8

                # Encoder Parameters
                self.COUNTS_PER_REV = 1000/3
                self.CM_PER_REV = 2 * math.pi * 30.0

                # Encoder variables for speed calculation
                self.ENCODER1_LAST_PULSE = time.time()
                self.motor1_speed = 0
                self.ENCODER2_LAST_PULSE = time.time()
                self.motor2_speed = 0

                GPIO.setmode(GPIO.BCM)

                # Pin Modes

                GPIO.setup(self.ENCODER1CHANNELA, GPIO.IN)
                # GPIO.setup(self.ENCODER1CHANNELB, GPIO.IN)
                GPIO.setup(self.ENCODER2CHANNELA, GPIO.IN)
                # GPIO.setup(self.ENCODER2CHANNELB, GPIO.IN)
                
                # GPIO interrupts to read encoder RISING edges
                GPIO.add_event_detect(self.ENCODER1CHANNELA, GPIO.RISING, callback=self.callback, bouncetime=50)
                GPIO.add_event_detect(self.ENCODER2CHANNELA, GPIO.RISING, callback=self.callback, bouncetime=50)


        # Encoder callback function on RISING edge
        def callback(self, channel):
                if channel == self.ENCODER1CHANNELA and GPIO.input(self.ENCODER1CHANNELA) == GPIO.HIGH:
                        deltaTime = time.time() - self.ENCODER1_LAST_PULSE
                        self.ENCODER1_LAST_PULSE = time.time()
                        encoder1countsPerSec = 1 / deltaTime
                        encoder1revsPerSec = encoder1countsPerSec / self.COUNTS_PER_REV
                        self.motor1_speed = encoder1revsPerSec * self.CM_PER_REV
                        print("Motor 1 Speed: " + str(self.motor1_speed))

                elif channel == self.ENCODER2CHANNELA and GPIO.input(self.ENCODER2CHANNELA) == GPIO.HIGH:
                        deltaTime = time.time() - self.ENCODER2_LAST_PULSE
                        self.ENCODER2_LAST_PULSE = time.time()
                        encoder2countsPerSec = 1 / deltaTime
                        encoder2revsPerSec = encoder2countsPerSec / self.COUNTS_PER_REV
                        self.motor2_speed = encoder2revsPerSec * self.CM_PER_REV
                        print("Motor 2 Speed: " + str(self.motor2_speed))
