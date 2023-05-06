#! /usr/bin/env python

import roslib
roslib.load_manifest('speed_tracker')
import rospy
import actionlib

from std_msgs.msg import Float32
from speed_tracker.msg import LaunchRoboCopAction, LaunchRoboCopGoal

THRESHOLD = 0.4


def feedback_log(feedback):
    rospy.loginfo(f"Distance: {feedback.distance}")
    rospy.loginfo(f"Deviation: {feedback.deviation}")
    rospy.loginfo(f"Speed left: {feedback.speed_left}")
    rospy.loginfo(f"Speed right: {feedback.speed_right}")

    
def check_speed(data):
    speed = data.data
    rospy.loginfo(speed)
    goal = LaunchRoboCopGoal()
    goal.time_stop = 5
    if(speed>THRESHOLD):
        client.send_goal(goal,feedback_cb=feedback_log)
        client.wait_for_result()
        time_chase = client.get_result()
        rospy.loginfo(f"Done heheheha: {time_chase}")

    

if __name__ == '__main__':
    rospy.init_node('robo_cop_client')
    client = actionlib.SimpleActionClient('robo_cop', LaunchRoboCopAction)
    rospy.loginfo("waiting for server")
    client.wait_for_server()
    rospy.Subscriber("/speed_tracker/velocity", Float32, check_speed)
    rospy.spin()

    