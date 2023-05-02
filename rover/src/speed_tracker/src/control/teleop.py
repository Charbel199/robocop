#! /usr/bin/python3.8

from pynput import keyboard
import rospy

from speed_tracker.msg import teleop_robocop, fuzzy_inputs

is_robocop = True

up_pressed = False
down_pressed = False
right_pressed = False
left_pressed = False

speed_left = 0
speed_right = 0
direction_left = 'forward'
direction_right = 'forward'

distance = 0
deviation = 0

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

    pub = rospy.Publisher(f"/robo_cop/{'teleop' if is_robocop else 'fuzzy_inputs'}", teleop_robocop if is_robocop else fuzzy_inputs, queue_size=10)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if is_robocop:
                if up_pressed and right_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 40, 20
                elif up_pressed and left_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 20, 40
                elif down_pressed and right_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 40, 20
                elif down_pressed and left_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 20, 40
                elif up_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 40, 40
                elif down_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'backward', 'backward', 40, 40
                elif right_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'backward', 'forward', 40, 40
                elif left_pressed:
                    direction_right, direction_left, speed_left, speed_right = 'forward', 'backward', 40, 40
                else:
                    direction_right, direction_left, speed_left, speed_right = 'forward', 'forward', 0, 0
                r.sleep()
                msg = teleop_robocop(direction_right=direction_right, direction_left=direction_left, speed_left=speed_left, speed_right = speed_right)
                pub.publish(msg)
            else:
                if up_pressed and distance < 2:
                        distance += 0.05
                if down_pressed and distance > 0:
                        distance -= 0.05
                if right_pressed and deviation < 20:
                        deviation += 0.5
                if left_pressed and deviation > -20:
                        deviation -= 0.5
                if not (up_pressed or down_pressed or right_pressed or left_pressed):
                    if distance > 0:
                        distance -= 0.05
                    if deviation > 0:
                        deviation -= 0.5
                    elif deviation < 0:
                        deviation += 0.5
                r.sleep()
                msg = fuzzy_inputs(distance=distance, deviation=deviation)
                pub.publish(msg)
