#! /usr/bin/env python

import RPi.GPIO as GPIO
import time


class Motor:
    def __init__(self, motor):
        GPIO.setmode(GPIO.BCM)

        # Motors' GPIO Pins
        MOTOR1_IN1_PIN = 5
        MOTOR1_IN2_PIN = 6
        MOTOR1_ENABLE_PIN = 16
        MOTOR2_IN1_PIN = 13
        MOTOR2_IN2_PIN = 19
        MOTOR2_ENABLE_PIN = 20


        # Setting Pin Modes and Initializing motors
        if motor==1:
            GPIO.setup(MOTOR1_ENABLE_PIN, GPIO.OUT)
            GPIO.setup(MOTOR1_IN1_PIN, GPIO.OUT)
            GPIO.setup(MOTOR1_IN2_PIN, GPIO.OUT)

            self.motor_pin1 = MOTOR1_IN1_PIN
            self.motor_pin2 = MOTOR1_IN2_PIN
            self.motor1_pwm = GPIO.PWM(MOTOR1_ENABLE_PIN, 1000)
            self.motor1_pwm.start(0)

        elif motor==2:
            GPIO.setup(MOTOR2_ENABLE_PIN, GPIO.OUT)
            GPIO.setup(MOTOR2_IN1_PIN, GPIO.OUT)
            GPIO.setup(MOTOR2_IN2_PIN, GPIO.OUT)

            self.motor_pin1 = MOTOR2_IN1_PIN
            self.motor_pin2 = MOTOR2_IN2_PIN
            self.motor2_pwm = GPIO.PWM(MOTOR2_ENABLE_PIN, 1000)
            self.motor2_pwm.start(0)
    
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
        self.motor1_pwm.ChangeDutyCycle(speed)

if __name__ == "__main__":
    motor1 = Motor(1)
    motor2 = Motor(2)
    time.sleep(5)

    motor1.set_motor("forward", 100) 
    motor2.set_motor("forward", 100) 
    time.sleep(3)

    motor1.set_motor("reverse", 100) 
    motor2.set_motor("reverse", 100) 
    time.sleep(3)

    motor1.set_motor("stop", 100) 
    motor2.set_motor("stop", 100) 
    GPIO.cleanup()
