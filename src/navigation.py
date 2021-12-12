# ******** IMPORTS ********
import time
import numpy as np
import math as m
import cv2
import Kalman

import utils
import local_navigation
from A_star_algorithm import run_A_Star

# ******** CONSTANTS ********
EPS_GOAL = 1
STEP_SKIP = 2  # number of nodes to skip when looking for the next node. Makes the navigation smoother.


# ******** FUNCTIONS ********
def next_node(myRobot, path):
    """
    Set the next node to visit

    :param:
        thymio: our robot
        path: optimal path returned from A* as a list of nodes.
        next node. makes the navigation smoother.
    :return:
        next_node: optimal next node (x,y) to visit
    """

    # Compute distance between Thymio and each position of the optimal path
    dist = np.linalg.norm(myRobot.get_pos() - path , axis=1)
    # Selects the closest point
    step = np.argmin(dist)
    if step + STEP_SKIP < len(path):
        _next = path[step + STEP_SKIP]
    else:
        _next = path[-1]
    _next=_next+[0.5,0.5]
    return _next


def go_to(myRobot, next_pos):
    """
    adjust the orientation of the robot by rotating him towards his targeted node then go forward
    :param:
        thymio: our robot
        next_node: targeted node to visit next.
    """
    target_vector = next_pos - myRobot.get_pos()  # vector between actual node and next node.
    theta_ref = m.atan2(target_vector[1], target_vector[0])  # Vector angle from x-axis
    dtheta = utils.normalise_angle(theta_ref - myRobot.get_angle())  # Normalize angle to minimise rotation

    if abs(dtheta) > m.pi / 16:
        theta = myRobot.get_angle() + dtheta

        if dtheta < 0:  # Opposite from convention
            myRobot.motor_rotate("left")
        else:
            myRobot.motor_rotate("right")
        time.sleep(abs(utils.FULLROTATIONTIME * dtheta / (2 * m.pi)))
        myRobot.motor_stop()
        myRobot.set_theta(theta)
    myRobot.motor_forward()


def callKalman(myRobot, Vision, Ts):
    """
    call Kalman and update our robot's position
    :param myRobot: our robot
    :param Vision: our vision agent
    :param speed: speed of the motors of our robot
    :param Ts: time spent between 2 calls of Kalman
    """
    Vision.read_image()
    #cv2.imwrite("image_end.png", Vision.image)
    robot_detected=Vision.get_robot()
    Vision.print_path()
    cv2.imwrite("image_end.png", Vision.image)
    vision_measure = [Vision.center_robot[0], Vision.center_robot[1], utils.deg_to_rad(Vision.angle)]

    x, y, theta = Kalman.kalman_filter(myRobot.get_previous_pos(), vision_measure, Ts, utils.MOTORSPEED,robot_detected)

    myRobot.set_pos(x, y)
    myRobot.set_theta(theta)


def follow_path(myRobot, Vision, path, goal):
    """
    Follow the path util the goal is reached and stop
    :param myRobot: our robot
    :param Vision:  our vision agent
    :param path: the path to follow
    :param goal: the goal
    """
    last_state = "global"  # robot state "global" or "local"
    next_pos = path[STEP_SKIP]

    # Adjust angle and goes forward
    go_to(myRobot, next_pos)
    # Starts timer
    t_start = time.perf_counter()
    time.sleep(0.5)
    # Keep going while goal not reached
    while np.linalg.norm(myRobot.get_pos() - goal) > EPS_GOAL:
        prox_sens = myRobot.th["prox.horizontal"]

        # Checking if should be in local or global navigation
        if sum([prox_sens[i] > local_navigation.THRESHOLD_OBST for i in range(0, 5)]) < 1:
            state = "global"  # global navigation

        else:
            state = "local"  # obstacle avoidance

        # Global navigation
        if state == "global":

            if last_state == "global":
                t_stop = time.perf_counter()
                Ts = t_stop - t_start
                # estimate new position
                callKalman(myRobot, Vision, Ts)

            next_pos = next_node(myRobot, path)

            # adjusts the orientation then goes forward
            go_to(myRobot, next_pos)

            # Restarts timer
            t_start = time.perf_counter()
            time.sleep(0.1)
            last_state = "global"

        # Local navigation
        elif state == "local":
            # Stops
            myRobot.motor_stop()

            # If previously in global navigation, computes actual position
            if last_state == "global":
                # Records time forward in global before entering in local
                t_stop = time.perf_counter()
                Ts = t_stop - t_start
                # estimate new position
                callKalman(myRobot, Vision, Ts)

            # Enters local navigation
            Ts = local_navigation.obstacle_avoidance(myRobot, prox_sens)

            # estimate new position
            callKalman(myRobot, Vision, Ts)

            # Reconstruct the path to the goal
            path, visitedNodes = run_A_Star(Vision.parcours, myRobot.get_pos(), goal)
            Vision.path=path
            last_state = "local"

    print("Goal reached")
    myRobot.motor_stop()
