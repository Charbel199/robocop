#! /usr/bin/env python

import roslib
roslib.load_manifest('speed_tracker')
import rospy
import actionlib
import math
import os
import sys
import RPi.GPIO as GPIO

from std_msgs.msg import Float32
from speed_tracker.msg import LaunchRoboCopAction, LaunchRoboCopResult, LaunchRoboCopFeedback

from control.fuzzy import FuzzyRoverController
from control.robocop_motor import *
from control.encoder import *

class RoboCopServer:
  def __init__(self):
    self.server = actionlib.SimpleActionServer('robo_cop', LaunchRoboCopAction, self.execute, False)
    self.feedback = LaunchRoboCopFeedback()


    self.RATE = 10
    self.r = rospy.Rate(self.RATE) # 1- Hz

    # initialzie fuzzy controller
    self.fuzzy_rover_controller = FuzzyRoverController()
    self.fuzzy_rover_controller.define_rules()
    self.fuzzy_rover_controller.create_control_system()

    # initialize 2 motors (1: left, 2: right)
    self.motor1 = Motor(1)
    self.motor2 = Motor(2)

    # initialize encoder for motor speeds
    self.encoder = Encoder()

    # fuzzy logic inputs
    rospy.Subscriber("/robo_cop/distance", Float32, self.check_distance)
    rospy.Subscriber("/robo_cop/deviation", Float32, self.check_deviation)
    self.distance = 0
    self.deviation = 0

    self.timer_chase = 0
    self.timer_stop = 0
    self.ZERO_THRESHOLD = 0.01
    self.server.start()

  def check_distance(self, data):
    self.distance = data.data
  
  def check_deviation(self, data):
    self.deviation = data.data


  def execute(self, goal):
    rospy.loginfo(f"Goal time is: {goal.time_stop}")

    while not rospy.is_shutdown():
      self.timer_chase += 1/self.RATE

      ##TODO get distance and thetta ...

      # Use fuzzy controller to actuate motors
      l_motor_value, r_motor_value = self.fuzzy_rover_controller.compute_output(self.distance, self.deviation, self.encoder.motor2_speed, self.encoder.motor1_speed)
      self.motor1.set_motor("forward", l_motor_value)
      self.motor2.set_motor("forward", r_motor_value)

      # return speed values as feedback
      speed1 = self.encoder.motor1_speed
      speed2 = self.encoder.motor2_speed
      speed_magnitude = math.sqrt(speed1**2 + speed2**2)

      if speed_magnitude<self.ZERO_THRESHOLD:
        self.timer_stop += 1/self.RATE
        if self.timer_stop >= goal.time_stop:
          result = LaunchRoboCopResult()
          result.time_chase = self.timer_chase
          self.server.set_succeeded(result)
          break
      else:
        self.timer_stop = 0

      self.feedback.distance = distance
      self.feedback.deviation = deviation
      self.feedback.speed_left = speed1
      self.feedback.speed_right = speed2

      self.server.publish_feedback(self.feedback)

      self.r.sleep()


    

if __name__ == '__main__':
  rospy.init_node('robo_cop_server')
  server = RoboCopServer()
  print("Started RoboCop Action Server")
  rospy.spin()
  GPIO.cleanup()
