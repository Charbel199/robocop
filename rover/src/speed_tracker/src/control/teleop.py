#! /usr/bin/python3.8

from pynput import keyboard
import rospy

from speed_tracker.msg import teleop

up_pressed = False
down_pressed = False
right_pressed = False
left_pressed = False

speed_left = 0
speed_right = 0
direction_left = 'forward'
direction_right = 'forward'

def on_press(key):
    global up_pressed, down_pressed, right_pressed, left_pressed
    if key == keyboard.Key.up:
        up_pressed = True
    if key == keyboard.Key.down:
        down_pressed = True
    if key == keyboard.Key.right:
        right_pressed = True
    if key == keyboard.Key.left:
        left_pressed = True

def on_release(key):
    global up_pressed, down_pressed, right_pressed, left_pressed
    if key == keyboard.Key.up:
        up_pressed = False
    if key == keyboard.Key.down:
        down_pressed = False
    if key == keyboard.Key.right:
        right_pressed = False
    if key == keyboard.Key.left:
        left_pressed = False


if __name__ == '__main__':
    rospy.init_node("robocop_teleop_node")
    rospy.loginfo("Teleop Node started.")
    pub = rospy.Publisher("/robocop_teleop", teleop, queue_size=10)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if up_pressed and right_pressed:
                direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 50, 30
            elif up_pressed and left_pressed:
                direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 30, 50
            elif down_pressed and right_pressed:
                direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 50, 30
            elif down_pressed and left_pressed:
                direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 30, 50
            elif up_pressed:
                direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 50, 50
            elif down_pressed:
                direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 50, 50
            elif right_pressed:
                direction_right, direction_left, speed_left, speed_right = 'backward', 'forward', 50, 50
            elif left_pressed:
                direction_right, direction_left, speed_left, speed_right = 'forward', 'backward', 50, 50
            else:
                direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 0, 0
            r.sleep()
            msg = teleop(direction_right=direction_right, direction_left=direction_left, speed_left=speed_left, speed_right = speed_right)
            pub.publish(msg)