# ******** IMPORTS ********
import numpy as np
import utils
import time
import math as m

# ******** CONSTANTS ********

THRESHOLD_SIDE = 100  # threshold setting the distance to the wall before going away of it
THRESHOLD_OBST = 2250  # threshold for entering in local navigation
TIME_FORWARD = 6  # time in seconds going away from obstacle before new computation


# ******** FUNCTIONS ********

def obstacle_avoidance(myRobot, prox_sens):
    """
    The robot avoid the obstacle by turning on himself until the obstacle is not detected
    then goes forward
    :param
        myRobot: our robot
        prox_sens: values of the proximity sensors
    :return:
        Ts: time spent going forward
        speed: average speed of the robot while going forward
    """
    # The obstacle is on the left - > rotate right
    if (prox_sens[0] + prox_sens[1]) > (prox_sens[4] + prox_sens[3]):
        Ts, speed = bypass("right", prox_sens, myRobot)
    # The obstacle is on the right - > rotate left
    else:
        Ts, speed = bypass("left", prox_sens, myRobot)

    return Ts, speed


def bypass(direction, prox_sens, myRobot):
    """
    Bypass the obstacle by turning on himself until the obstacle is not detected
    then goes forward
    :param
        direction: direction of the rotation
         prox_sens: values of the proximity sensors
        myRobot: our robot
    :return:
        Ts: time spent going forward
        speed: average speed of the robot while going forward
    """
    reset_timer = 1  # to start the timer only the first time one enter into the loop
    # the robot turn on himself until the sensor don't detect the obstacle
    while sum(prox_sens[i] > THRESHOLD_SIDE for i in range(0, 5)) > 0:

        myRobot.motor_rotate(direction)

        if reset_timer:
            t_start = time.perf_counter()
            reset_timer = 0

        prox_sens = myRobot.th["prox.horizontal"]
        if sum(prox_sens[i] > THRESHOLD_SIDE for i in range(0, 5)) == 0:
            # Stop the timer
            t_stop = time.perf_counter()
            turning_time = t_stop - t_start
            # Computes the angle variation
            dtheta = turning_time * 2 * m.pi / utils.FULLROTATIONTIME
            # Update theta
            if direction == "right":  # Opposite from convention
                theta = myRobot.get_angle() + dtheta
            else:
                theta = myRobot.get_angle() - dtheta

            myRobot.set_theta(theta)

            # Go forward
            Ts, speed = local_forward(myRobot)

    return Ts, speed


def local_forward(myRobot):
    """
    Go forward for TIME_FORWARD seconds or until an obstacle is detected
    :param myRobot: our robot
    :return:
        Ts: time spent going forward
        speed: average speed of the robot while going forward
    """
    # Set motors speed to go forward
    speed = 0
    myRobot.motor_forward()

    # Increment time counter while checking for obstacles
    step_forward = TIME_FORWARD / 50
    nb_step = 0
    for i in range(50):
        prox = myRobot.th["prox.horizontal"]
        # Break if an obstacle is detected
        if sum(prox[i] > THRESHOLD_SIDE for i in range(0, 5)):
            break
        time.sleep(step_forward)
        speed = speed + np.mean(myRobot.get_speed())
        nb_step = nb_step + 1

    myRobot.motor_stop()

    # Compute time forward and average speed
    Ts = nb_step * step_forward
    if nb_step != 0:
        speed = speed / nb_step

    return Ts, speed
