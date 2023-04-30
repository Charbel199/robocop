#! /usr/bin/env python

import RPi.GPIO as GPIO
import time
import rospy

from speed_tracker.msg import teleop

class Motor:
    def __init__(self, motor, teleop=False):
        GPIO.setmode(GPIO.BCM)
        self.motor = motor
        if teleop:
            rospy.Subscriber("/robocop_teleop", teleop, self.cb)

        # Motors' GPIO Pins
        MOTOR1_IN1_PIN = 5
        MOTOR1_IN2_PIN = 6
        MOTOR1_ENABLE_PIN = 16
        MOTOR2_IN1_PIN = 19
        MOTOR2_IN2_PIN = 13
        MOTOR2_ENABLE_PIN = 20


        # Setting Pin Modes and Initializing motors
        if motor==1:
            GPIO.setup(MOTOR1_ENABLE_PIN, GPIO.OUT)
            GPIO.setup(MOTOR1_IN1_PIN, GPIO.OUT)
            GPIO.setup(MOTOR1_IN2_PIN, GPIO.OUT)

            self.motor_pin1 = MOTOR1_IN1_PIN
            self.motor_pin2 = MOTOR1_IN2_PIN
            self.motor_pwm = GPIO.PWM(MOTOR1_ENABLE_PIN, 1000)
            self.motor_pwm.start(0)

        elif motor==2:
            GPIO.setup(MOTOR2_ENABLE_PIN, GPIO.OUT)
            GPIO.setup(MOTOR2_IN1_PIN, GPIO.OUT)
            GPIO.setup(MOTOR2_IN2_PIN, GPIO.OUT)

            self.motor_pin1 = MOTOR2_IN1_PIN
            self.motor_pin2 = MOTOR2_IN2_PIN
            self.motor_pwm = GPIO.PWM(MOTOR2_ENABLE_PIN, 1000)
            self.motor_pwm.start(0)
    
    def cb(self, data):
        if self.motor == 1:
            self.set_motor(data.direction_left, data.speed_left)
        else:
            self.set_motor(data.direction_right, data.speed_right)

    def set_motor(self, direction, speed):
        if direction == "forward":
            GPIO.output(self.motor_pin1, GPIO.HIGH)
            GPIO.output(self.motor_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(self.motor_pin1, GPIO.LOW)
            GPIO.output(self.motor_pin2, GPIO.HIGH)
        elif direction == "stop":
            GPIO.output(self.motor_pin1, GPIO.LOW)
            GPIO.output(self.motor_pin2, GPIO.LOW)
        else:
            GPIO.output(self.motor_pin1, GPIO.HIGH)
            GPIO.output(self.motor_pin2, GPIO.HIGH)
        self.motor_pwm.ChangeDutyCycle(speed)

if __name__ == "__main__":
    rospy.init_node("robocop_motors")
    motor1 = Motor(1, True)
    motor2 = Motor(2, True)
    rospy.spin()
    GPIO.cleanup()
