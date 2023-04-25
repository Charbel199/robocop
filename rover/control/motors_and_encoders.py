import RPi.GPIO as GPIO
import time
import math

# Motors' GPIO Pins
motor1_in1_pin = 5
motor1_in2_pin = 6
motor1_enable_pin = 16
motor2_in1_pin = 13
motor2_in2_pin = 19
motor2_enable_pin = 20

# Encoder GPIO Pins
encoder1channelA = 9
encoder1channelB = 11
encoder2channelA = 25
encoder2channelB = 8

# Encoder Parameters
countsPerRev = 1000/3
cmPerRev = 2 * math.pi * 30.0

# Encoder variables for speed calculation
encoder1LastPulse = time.time()
encoder1cmPerSec = 0
encoder2LastPulse = time.time()
encoder2cmPerSec = 0

GPIO.setmode(GPIO.BCM)

# Pin Modes
GPIO.setup(motor1_enable_pin, GPIO.OUT)
GPIO.setup(motor2_enable_pin, GPIO.OUT)

GPIO.setup(motor1_in1_pin, GPIO.OUT)
GPIO.setup(motor1_in2_pin, GPIO.OUT)
GPIO.setup(motor2_in1_pin, GPIO.OUT)
GPIO.setup(motor2_in2_pin, GPIO.OUT)

GPIO.setup(encoder1channelA, GPIO.IN)
GPIO.setup(encoder1channelB, GPIO.IN)
GPIO.setup(encoder2channelA, GPIO.IN)
GPIO.setup(encoder2channelB, GPIO.IN)

# Initializing motors
motor1_pwm = GPIO.PWM(motor1_enable_pin, 1000)
motor2_pwm = GPIO.PWM(motor2_enable_pin, 1000)
motor1_pwm.start(0)
motor2_pwm.start(0)

# Encoder callback function on RISING edge
def callback(channel):
    global encoder1LastPulse, encoder1cmPerSec, encoder2LastPulse, encoder2cmPerSec
    if channel == encoder1channelA and GPIO.input(encoder1channelA) == GPIO.HIGH:
            deltaTime = time.time() - encoder1LastPulse
            encoder1LastPulse = time.time()
            encoder1countsPerSec = 1 / deltaTime
            encoder1revsPerSec = encoder1countsPerSec / countsPerRev
            encoder1cmPerSec = encoder1revsPerSec * cmPerRev
            print("Motor 1 Speed: " + str(encoder1cmPerSec))
    elif channel == encoder2channelA and GPIO.input(encoder2channelA) == GPIO.HIGH:
            deltaTime = time.time() - encoder2LastPulse
            encoder2LastPulse = time.time()
            encoder2countsPerSec = 1 / deltaTime
            encoder2revsPerSec = encoder2countsPerSec / countsPerRev
            encoder2cmPerSec = encoder2revsPerSec * cmPerRev
            print("Motor 2 Speed: " + str(encoder2cmPerSec))

# Motor control function using direction and duty cycle
def set_motor(motor, direction, speed):
    if motor == 1:
        if direction == "forward":
            GPIO.output(motor1_in1_pin, GPIO.HIGH)
            GPIO.output(motor1_in2_pin, GPIO.LOW)
        elif direction == "reverse":
            GPIO.output(motor1_in1_pin, GPIO.LOW)
            GPIO.output(motor1_in2_pin, GPIO.HIGH)
        else:
            GPIO.output(motor1_in1_pin, GPIO.LOW)
            GPIO.output(motor1_in2_pin, GPIO.LOW)
        motor1_pwm.ChangeDutyCycle(speed)
    elif motor == 2:
        if direction == "forward":
            GPIO.output(motor2_in1_pin, GPIO.HIGH)
            GPIO.output(motor2_in2_pin, GPIO.LOW)
        elif direction == "reverse":
            GPIO.output(motor2_in1_pin, GPIO.LOW)
            GPIO.output(motor2_in2_pin, GPIO.HIGH)
        else:
            GPIO.output(motor2_in1_pin, GPIO.LOW)
            GPIO.output(motor2_in2_pin, GPIO.LOW)
        motor2_pwm.ChangeDutyCycle(speed)

# GPIO interrupts to read encoder RISING edges
GPIO.add_event_detect(encoder1channelA, GPIO.RISING, callback=callback, bouncetime=50)
GPIO.add_event_detect(encoder2channelA, GPIO.RISING, callback=callback, bouncetime=50)

if __name__ == "__main__":
    time.sleep(5)

    set_motor(1, "forward", 100) 
    set_motor(2, "forward", 100) 
    time.sleep(3)

    set_motor(1, "reverse", 100)
    set_motor(2, "reverse", 100)
    time.sleep(3)

    set_motor(1, "stop", 0)
    set_motor(2, "stop", 0)
    GPIO.cleanup()