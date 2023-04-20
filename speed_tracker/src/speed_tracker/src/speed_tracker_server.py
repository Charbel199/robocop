#! /usr/bin/env python

import roslib
roslib.load_manifest('speed_tracker')
import rospy
import actionlib

from speed_tracker.msg import LaunchRoboCopAction, LaunchRoboCopResult

class RoboCopServer:
  def __init__(self):
    self.server = actionlib.SimpleActionServer('robo_cop', LaunchRoboCopAction, self.execute, False)
    self.server.start()

  def execute(self, goal):
    rospy.loginfo(f"Goal time is: {goal.time_stop}")
    # Do lots of awesome groundbreaking robot stuff here
    result = LaunchRoboCopResult()
    result.time_chase = goal.time_stop + 5
    self.server.set_succeeded(result)


if __name__ == '__main__':
  rospy.init_node('robo_cop_server')
  server = RoboCopServer()
  rospy.spin()