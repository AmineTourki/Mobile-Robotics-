import math as m
import navigation

OBSTACLE_TH_MIN = 2500
OBSTACLE_TH_MAX = 5000
ROTATION_FRONT = m.pi / 3
ROTATION_MIDDLE = m.pi / 4
ROTATION_SIDE = m.pi / 9
LOCAL_FORWARD = 0.2  # seconds


def detect_obstacle(prox_sensors):
    """
    Detect obstacle
    :param prox_sensors: proximity sensors values (from sensor 0 to 4)
    :return: True if obstacle detected, False otherwise
    """
    for sensor in prox_sensors:
        if sensor > OBSTACLE_TH_MIN:
            return True
    return False


def correct_sensor_val(sensor_val):
    """
    correct sensor value to avoid values out of the expected range.
    :param sensor_val: value of the proximity sensor
    :return: sensor_val: corrected value
    """
    # correct sensor val to avoid division by zero and false rotation angles
    if sensor_val <= 0:
        sensor_val = 1
    elif sensor_val > OBSTACLE_TH_MAX:
        sensor_val = OBSTACLE_TH_MAX
    return sensor_val


def sensor_to_angle(side_sensor, middle_sensor, front_sensor):
    """
    calculate the angle by which the robot should rotate for one side
    (right or left)
    :param side_sensor: value of the sensor on the side
    :param middle_sensor: value of the sensor on the front
    :return: rotation angle in radians
    """
    side_sensor = correct_sensor_val(side_sensor)
    middle_sensor = correct_sensor_val(middle_sensor)
    front_sensor = correct_sensor_val(front_sensor)

    sum_sensors = side_sensor + middle_sensor + front_sensor
    return side_sensor / sum_sensors * ROTATION_SIDE + middle_sensor / sum_sensors * ROTATION_MIDDLE + front_sensor / sum_sensors * ROTATION_FRONT


def rotation_angle(prox_sensors):
    """
    calculate the angle by which to rotate the robot
    :param prox_sensors: sensors values (from sensor 0 to 4)
    :return: rotation angle in radians
    """
    angle_right = sensor_to_angle(prox_sensors[4], prox_sensors[3], prox_sensors[2])
    angle_left = sensor_to_angle(prox_sensors[0], prox_sensors[1], prox_sensors[2])
    if prox_sensors[0] + prox_sensors[1] > prox_sensors[3] + prox_sensors[4]:
        return angle_left
    else:
        return -angle_right


def avoid_obstacle(myRobot, Vision):
    """
    main function in local navigation to avoid obstacles
    :param myRobot: robot instance
    :param Vision: vision instance
    :return: True if local navigation occurs, False otherwise
    """
    prox_sensors = myRobot.th["prox.horizontal"]
    if not detect_obstacle(prox_sensors):
        return False
    while detect_obstacle(prox_sensors):
        myRobot.motor_stop()
        angle = rotation_angle(prox_sensors)
        myRobot.motor_rotate(angle)
        myRobot.motor_forward(LOCAL_FORWARD)
        navigation.callKalman(myRobot, Vision, LOCAL_FORWARD)
        prox_sensors = myRobot.th["prox.horizontal"]
    return True
