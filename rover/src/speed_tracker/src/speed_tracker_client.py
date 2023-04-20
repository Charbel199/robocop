#! /usr/bin/env python

import roslib
roslib.load_manifest('speed_tracker')
import rospy
import actionlib

from std_msgs.msg import Float32
from speed_tracker.msg import LaunchRoboCopAction, LaunchRoboCopGoal

THRESHOLD = 3

    
def check_speed(data):
    speed = data.data
    rospy.loginfo(speed)
    goal = LaunchRoboCopGoal()
    goal.time_stop = 5.3
    if(speed>THRESHOLD):
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(5.0))
        time_chase = client.get_result()
        rospy.loginfo(f"Done heheheha: {time_chase}")

    

if __name__ == '__main__':
    rospy.init_node('robo_cop_client')
    client = actionlib.SimpleActionClient('robo_cop', LaunchRoboCopAction)
    rospy.loginfo("waiting for server")
    client.wait_for_server()
    rospy.Subscriber("/speed_tracker/velocity", Float32, check_speed)
    rospy.spin()

    