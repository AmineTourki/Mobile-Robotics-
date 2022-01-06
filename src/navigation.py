# ******** IMPORTS ********
import numpy as np
import math
import Kalman
import utils
import local_navigation
import cv2

# ******** CONSTANTS ********
EPS_GOAL = 1.5
GLOBAL_FORWARD = 0.2  # step time (in seconds) spent going forward in global navigation
# ******** FUNCTIONS ********

def callKalman(myRobot, Vision, time_forward):
    """
    call Kalman and update our robot's position
    :param myRobot: our robot, instance of Class MyThymio
    :param Vision: our vision agent, instance of Class Vision_Agent
    :param time_forward: step time (in seconds) spent going forward
    """
    # read and save image
    Vision.read_image()
    robot_detected = Vision.get_robot()
    Vision.print_path()
    cv2.imwrite("image_end.png", Vision.image)

    vision_measure = [Vision.center_robot[0], Vision.center_robot[1], utils.deg_to_rad(Vision.angle)]

    x, y, theta = Kalman.kalman_filter(myRobot.get_previous_pos(), vision_measure, time_forward, utils.MOTORSPEED,
                                       robot_detected)

    myRobot.set_pos(x, y)
    myRobot.set_angle(theta)


def closest_node_index(current_pos, path, old_index):
    """
    return the closest node of the path to the current position of the robot
    and if the robot should change his orientation

    :param current_pos: current position of the robot
    :param path: array containing the optimal path to follow
    :param old_index: index taken in the last call of this function
    :return:    closestnode_index (int)
                change_angle (bool)

    """
    # check the nodes from old index and ahead and get the index of the closest node to our current position
    closestnode_index = old_index + np.linalg.norm(current_pos - path[old_index:], axis=1).argmin()
    # only change the orientation if the robot is closer to his next node compared to the initial node
    # don't change angle if already close to goal (goal index ==len(path) - 1)
    change_angle = (closestnode_index != old_index) and (closestnode_index != len(path) - 1)
    return change_angle, closestnode_index


def change_orientation(myRobot, next_pos):
    """
    adjust the orientation of the robot by rotating him towards his next_pos
    :param myRobot: our robot, instance of Class MyThymio
    :param next_pos: coordinate of the next position
    """
    target_vector = next_pos - myRobot.get_pos()
    dtheta = math.atan2(target_vector[1], target_vector[0]) - myRobot.get_angle()
    # rotate the robot with angle dtheta
    myRobot.motor_rotate(dtheta)


def follow_path(myRobot, Vision, path):
    """
    Follow the path until the goal is reached or until the robot enters local navigation
    :param myRobot: our robot, instance of Class MyThymio
    :param Vision: our vision agent, instance of Class Vision_Agent
    :param path: array containing the optimal path to follow
    :return:    True if the goal is reached
                False if the robot entered local navigation (goal not reached)
    """
    closest_index = 0
    first_step = 1
    while math.dist(myRobot.get_pos(), path[-1]) > EPS_GOAL:  # path[-1] is the goal
        change_angle, closest_index = closest_node_index(myRobot.get_pos(), path, closest_index)
        # only change the orientation if the robot is closest to his next node compared to the initial node
        if change_angle or first_step:
            first_step = 0
            next_pos = path[closest_index + 1]
            # rotate to face the new target node next_pos
            change_orientation(myRobot, next_pos)
        # go forward GLOBAL_FORWARD(0.2) seconds
        myRobot.motor_forward(GLOBAL_FORWARD)
        # estimate new position and update myRobot pose
        callKalman(myRobot, Vision, GLOBAL_FORWARD)
        # enter local navigation if obstacle detected
        entered_local_navigation = local_navigation.avoid_obstacle(myRobot, Vision)
        if entered_local_navigation:
            # if the robot entered local navigation , return to the main implementation loop to construct a new path
            print("Local navigation")
            return False

    print("Goal reached")
    myRobot.motor_stop()
    return True
