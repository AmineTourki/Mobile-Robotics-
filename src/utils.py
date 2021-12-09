# ******** IMPORTS ********
import numpy as np
import math as m

# ******** CONSTANTS ********

MOTORSPEED = 50  # Forward speed
MOTOR_ROT = 50  # Rotation speed
FULLROTATIONTIME = 16.5  # time in seconds to turn 2 pi rad


# ******** CLASSES ********


class MyThymio(object):
    """
    Class representing our Thymio robot, gathering state information and class
    methods to modifiy its state, command the motors, get Kalman filter values
    to adapt its motor speed

    """

    def __init__(self, th, x, y, theta):
        """
        Initialisation of the object
        :param th: instance linking to the Thymio robot connected via USB
               x,y: x and y positions of the center of the Thymio
               theta: the absolute angle orientation of the Thymio
        :return: creates the object of class MyThymio
        """

        self.th = th
        self.x = x
        self.y = y
        self.angle = theta
        self.inLocal = False
        self.reached = False

    def get_pos(self):
        return np.array([self.pos_x, self.pos_y])

    def set_pos(self, x, y):
        """
        called only after kalman
        """
        self.x = x
        self.y = y

    def motor_stop(self):
        """
        Set the speed of the motors to 0 to make the Thymio stop
        """
        self.th.set_var("motor.left.target", 0)
        self.th.set_var("motor.right.target", 0)

    def motor_forward(self):
        """
        Set the speed of both motors to MOTORSPEED to make the Thymio move forward
        """
        self.th.set_var("motor.left.target", MOTORSPEED)
        self.th.set_var("motor.right.target", MOTORSPEED)

    def motor_rotate(self, direction):
        """
        Set the speed of the motors to ±MOTOR_ROT to make the Thymio turn on himself
        :param direction: direction of the rotation
        """

        if direction == "left":
            self.th.set_var("motor.right.target", MOTOR_ROT)
            self.th.set_var("motor.left.target", 2 ** 16 - MOTOR_ROT)

        if direction == "right":
            self.th.set_var("motor.left.target", MOTOR_ROT)
            self.th.set_var("motor.right.target", 2 ** 16 - MOTOR_ROT)

    def get_angle(self):
        """
        Reads the absolute angle of Thymio
        :return: angle in rad
        """
        return self.angle

    def set_theta(self, theta):
        """
        set the rotation of the robot
        :param theta: the absolute angle of Thymio
        """
        self.angle = theta

    def get_speed(self):
        """
        Return the left and right speeds of the robot
        """
        speed_left = self.th["motor.left.speed"]
        speed_right = self.th["motor.right.speed"]
        return [speed_left, speed_right]

    def get_previous_pos(self):
        """
        Used in kalman.Since x and y are only updated just after calling kalman,
        they are the previous positions of our robot when calling it.
        """
        return np.array([self.x , self.y, self.angle])


def normalise_angle(alpha):
    """
    return the angle between ± pi
    :param alpha: the angle to normalise
    """
    while alpha > m.pi:
        alpha -= 2. * m.pi
    while alpha < -m.pi:
        alpha += 2. * m.pi
    return alpha


def deg_to_rad(alpha):
    return alpha * m.pi / 180

