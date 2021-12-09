#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 08:22:34 2021

@author: alexandre
"""



import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos
from sympy import init_printing



class Kalman_Agent:
    def __init__(self):
        self.numstates=5 # States
        self.dt = 1.0/50.0 # Sample Rate of the Measurements is 50Hz
        self.dtGPS=1.0/10.0 # Sample Rate of GPS is 10Hz
        vs, psis, dpsis, dts, xs, ys, lats, lons = symbols('v \psi \dot\psi T x y lat lon')
        
        self.gs = Matrix([[xs+(vs/dpsis)*(sin(psis+dpsis*dts)-sin(psis))],
                     [ys+(vs/dpsis)*(-cos(psis+dpsis*dts)+cos(psis))],
                     [psis+dpsis*dts],
                     [vs],
                     [dpsis]])
        self.state = Matrix([xs,ys,psis,vs,dpsis])
        self.P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        dt = self.dt
        wheels_dist=9.35
        sGPS     = 0.5*15*dt**2  # assume 15cm/s2 as maximum acceleration, forcing the vehicle
        sCourse  = (15/(wheels_dist/2))*dt # assume 0.1rad/s as maximum turn rate for the vehicle
        sVelocity= 15*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
        sYaw     = 0.5*(15/(wheels_dist/2))*dt**2 # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
        
        self.Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2])
        
        size_px_cm = 1
        varGPS = size_px_cm # Standard Deviation of GPS Measurement
        varspeed = 0.12 # Variance of the speed measurement
        varyaw = 0.12/(wheels_dist/2) # Variance of the yawrate measurement
        self.R = np.matrix([[varGPS**2, 0.0, 0.0, 0.0],
                       [0.0, varGPS**2, 0.0, 0.0],
                       [0.0, 0.0, varspeed**2, 0.0],
                       [0.0, 0.0, 0.0, varyaw**2]])
        
        hs = Matrix([[xs],
             [ys],
             [vs],
             [dpsis]])
        

        self.JHs=hs.jacobian(self.state)
        
        self.I = np.eye(self.numstates)

        

        # Preallocation for Plotting
        self.x0 = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.x5 = []
        self.Zx = []
        self.Zy = []
        self.Px = []
        self.Py = []
        self.Pdx= []
        self.Pdy= []
        self.Pddx=[]
        self.Pddy=[]
        self.Kx = []
        self.Ky = []
        self.Kdx= []
        self.Kdy= []
        self.Kddx=[]
        self.dstate=[]
        
        
    def savestates(self,x, Z, P, K):
        self.x0.append(float(x[0]))
        self.x1.append(float(x[1]))
        self.x2.append(float(x[2]))
        self.x3.append(float(x[3]))
        self.x4.append(float(x[4]))
        self.Zx.append(float(Z[0]))
        self.Zy.append(float(Z[1]))    
        self.Px.append(float(P[0,0]))
        self.Py.append(float(P[1,1]))
        self.Pdx.append(float(P[2,2]))
        self.Pdy.append(float(P[3,3]))
        self.Pddx.append(float(P[4,4]))
        self.Kx.append(float(K[0,0]))
        self.Ky.append(float(K[1,0]))
        self.Kdx.append(float(K[2,0]))
        self.Kdy.append(float(K[3,0]))
        self.Kddx.append(float(K[4,0]))
        
        
    def update(self,x,Z):
        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        # see "Dynamic Matrix"
        if np.abs(yawrate[filterstep])<0.0001: # Driving straight
            x[0] = x[0] + x[3]*dt * np.cos(x[2])
            x[1] = x[1] + x[3]*dt * np.sin(x[2])
            x[2] = x[2]
            x[3] = x[3]
            x[4] = 0.0000001 # avoid numerical issues in Jacobians
            self.dstate.append(0)
        else: # otherwise
            x[0] = x[0] + (x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2]))
            x[1] = x[1] + (x[3]/x[4]) * (-np.cos(x[4]*dt+x[2])+ np.cos(x[2]))
            x[2] = (x[2] + x[4]*dt + np.pi) % (2.0*np.pi) - np.pi
            x[3] = x[3]
            x[4] = x[4]
            self.dstate.append(1)
        
        # Calculate the Jacobian of the Dynamic Matrix A
        # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
        a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
        a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
        a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
        a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
        a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
        a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
        JA = np.matrix([[1.0, 0.0, a13, a14, a15],
                        [0.0, 1.0, a23, a24, a25],
                        [0.0, 0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])
        
        
        # Project the error covariance ahead
        self.P = JA*self.P*JA.T + self.Q
        
        # Measurement Update (Correction)
        # ===============================
        # Measurement Function
        hx = np.matrix([[float(x[0])],
                        [float(x[1])],
                        [float(x[3])],
                        [float(x[4])]])
        
        if GPS[filterstep]: # with 10Hz, every 5th step
            JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        else: # every other step
            JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])        
        
        S = JH*self.P*JH.T + self.R
        K = (self.P*JH.T) * np.linalg.inv(S)
        
        # Update the estimate via
        Z = measurements[:,filterstep].reshape(JH.shape[0],1)
        y = Z - (hx)                         # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        self.P = (self.I - (K*JH))*self.P
        
        
        
        # Save states for Plotting
        savestates(x, Z, self.P, K)
        return x

    
kalman = Kalman_Agent()



