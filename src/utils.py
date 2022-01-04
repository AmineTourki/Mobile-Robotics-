# ******** IMPORTS ********
import numpy as np
import math as m
import time

# ******** CONSTANTS ********

MOTORSPEED = 50
FULLROTATIONTIME = 16.5  # time in seconds to turn 2 pi rad

# ******** CLASS ********
class MyThymio(object):

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
        self.angle = normalise_angle(theta)

    def get_pos(self):
        return np.array([self.x, self.y])

    def set_pos(self, x, y):
        """
        Set the robot's position.Called only after kalman.
        """
        self.x = x
        self.y = y

    def motor_stop(self):
        """
        Set the speed of the motors to 0 to make the robot stop
        """
        self.th.set_var("motor.left.target", 0)
        self.th.set_var("motor.right.target", 0)

    def motor_forward(self, time_forward):
        """
        Set the speed of both motors to MOTORSPEED  for time_forward seconds
        """
        self.th.set_var("motor.left.target", val=MOTORSPEED)
        self.th.set_var("motor.right.target", val=MOTORSPEED)
        time.sleep(time_forward)
        self.motor_stop()

    def motor_rotate(self, dtheta):
        """
        Set the speed of the motors to ±MOTORSPEED to make the Thymio turn on himself dtheta rad
        :param dtheta: angle of rotation in rad
        """
        dtheta = normalise_angle(dtheta)
        if dtheta < 0:  # rotate left
            self.th.set_var("motor.right.target", val=MOTORSPEED)
            self.th.set_var("motor.left.target", val=2 ** 16 - MOTORSPEED)

        elif dtheta > 0:  # rotate right
            self.th.set_var("motor.left.target", val=MOTORSPEED)
            self.th.set_var("motor.right.target", val=2 ** 16 - MOTORSPEED)

        time.sleep(abs(FULLROTATIONTIME * dtheta / (2 * m.pi)))
        self.angle = normalise_angle(self.angle + dtheta)
        self.motor_stop()

    def get_angle(self):
        """
        Read the absolute angle of the robot
        :return: angle in rad
        """
        return self.angle

    def set_angle(self, theta):
        """
        set the absolute angle of the robot
        :param theta: the absolute angle of the robot
        """
        theta = normalise_angle(theta)
        self.angle = theta

    def get_previous_pos(self):
        """
        Used in kalman.Since x and y are only updated just after calling kalman,
        they are the previous positions of our robot when calling it.
        """
        return [self.x, self.y, self.angle]


# ******** Useful Functions ********
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
