import numpy as np
import math as m
import vision_agent

# speed conversion to go from Thymio speed to speed in cm/s
SPEED_CM_S_COEFF = 0.043478260869565216
NOD_SIZE_CM = 2
H = np.array([[1,0,0],[0,1,0],[0,0,1]])
R = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]])
Q = np.array([[0.12306,0,0],[0,0.12306,0],[0,0,m.pi/60]])
P = np.diag([100,100,m.pi/4])

def Kalman_update(X_est_priori, P_est_priori, X_measure):
    """
    Update step of the Kalman filter
    :param: X_est_priori: A priori estimation of the position
    :param: P_est_priori: A priori estimation of the covariance
    :param: X_measure: measured position by the camera
    :return: X_est_post with updated position and robot orientation
    :return: P_est_post updated covariance matrix
    """
    # converrt the values from nod to cm
    X_measure[0] = X_measure[0] * NOD_SIZE_CM
    X_measure[1] = X_measure[1] * NOD_SIZE_CM

    # innovation / measurement residual
    i = X_measure - np.dot(H,X_est_priori)
    # measurement prediction covariance
    S = np.dot(H, np.dot(P_est_priori, H.T)) + R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_priori, np.dot(H.T, np.linalg.inv(S)))

    # a posteriori estimate
    X_est_post = X_est_priori + np.dot(K, i)
    P_est_post = P_est_priori - np.dot(K, np.dot(H, P_est_priori))

    return X_est_post,P_est_post


# x_priori = x_previous + vitesse*time_sampling*cos(alpha)
# y_priori = x_previous + vitesse*time_sampling*sin(alpha)
# alpha_priori = alpha_previous
#
# A = np.array([[1,0,v*t_sampling*-sin(alpha)],
#             [0,1,v*t_sampling*cos(alpha)],
#             [0,0,1]])

def Kalman_estimate(X_previous,P_previous,delta_t,speed):
    """
    Estimation step of the Kalman filter
    :param: X_previous: previous position and orientation of the robot
    :param: P_previous: previous covariance matrix
    :param: delta_t: time passed between two calls of Kalman.
    :param: speed: speed of the robot
    :return: A priori position and covariance matrix of the robot
    """

    # convert speed from Thymio speed to cm/s
    speed = speed * SPEED_CM_S_COEFF
    # convert position from nod to cm
    X_previous[0] = X_previous[0]*NOD_SIZE_CM
    X_previous[1] = X_previous[1] * NOD_SIZE_CM
    # calculate the state matrix
    A = np.array([[1, 0, -speed * delta_t * m.sin(X_previous[-1])],
                  [0, 0, speed * delta_t * m.cos(X_previous[-1])],
                  [0,0,0]])

    X_priori = X_previous + np.array([speed * delta_t * m.cos(X_previous[-1]) , speed * delta_t * m.sin(X_previous[-1]) , 0])
    P_priori = np.dot(A, np.dot(P_previous, A.T)) + Q

    return X_priori,P_priori

def Kalman_filter(X_previous,P_previous,X_measure,delta_t,speed):
    """
    Kalman filter with estimation and update steps
    :param: X_previous: previous position and orientation of the robot
    :param: P_previous: previous covariance matrix
    :param: X_measure: current position measured with the camera
    :param: delta_t: time passed between two calls of Kalman.
    :param: speed: speed of the robot
    :return: position and orientation of the robot after correction
    """
    global P

    X_priori , P_priori = Kalman_estimate(X_previous, P_previous, delta_t, speed)
    X_est_post,P = Kalman_update(X_priori, P_priori, X_measure)

    # return the position in nod
    nod_x = int(X_est_post[0]/NOD_SIZE_CM)
    nod_y = int(X_est_post[1]/NOD_SIZE_CM)
    return nod_x, nod_y ,X_est_post[2]

def Kalman_init(nod_cm,pixel_cm):
    """
    initialise the movement covariance and size of pixel in cm
    :param nod_cm: size in cm of a cell in the map grid
    :param pixel_cm: size of a pixel in cm
    """
    global NOD_SIZE_CM
    global R

    NOD_SIZE_CM = nod_cm
    # uncertainty of 4 pixel
    R = np.array([[5*pixel_cm,0,0],[0,5*pixel_cm,0],[0,0,m.pi/40]])
