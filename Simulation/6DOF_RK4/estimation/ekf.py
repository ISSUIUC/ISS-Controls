# all the linear positions
from filterpy.common import Q_continuous_white_noise
import numpy as np
import util.vectors as vct

class KalmanFilter:
    """Kalman Filter for 3D position estimation
    
    Args:
        dt (float): time step
        pos_x (float): initial x position
        vel_x (float): initial x velocity
        accel_x (float): initial x acceleration
        pos_y (float): initial y position
        vel_y (float): initial y velocity
        accel_y (float): initial y acceleration
        pos_z (float): initial z position
        vel_z (float): initial z velocity
        accel_z (float): initial z acceleration

        Matrices:
            F (np.ndarray): state transition matrix
            H (np.ndarray): observation matrix
            Q (np.ndarray): process noise covariance matrix
            R (np.ndarray): measurement noise covariance matrix
            P_k (np.ndarray): current state covariance matrix
            x_k (np.ndarray): current state vector
            x_priori (np.ndarray): priori state vector
            P_priori (np.ndarray): priori covariance matrix
    """
    def __init__(self, dt, pos_x, vel_x, accel_x, pos_y, vel_y, accel_y, pos_z, vel_z, accel_z):
        self.dt = dt
        self.x_k = np.zeros((9,1))
        self.Q = np.zeros((9,9))
        self.R = np.diag([2., 1.9, 1.9, 1.9])
        self.P_k = np.eye(9)
        self.x_priori = np.zeros((9,1))
        self.P_priori = np.zeros((9,9))
        self.F = np.zeros((9,9))
        self.H = np.zeros((4,9))

        self.current_time = 0
        self.s_dt = dt

        self.x_k = np.array([pos_x, vel_x, accel_x, pos_y, vel_y, accel_y, pos_z, vel_z, accel_z]).T
        self.w_k = np.zeros((3))  # angular velocity, not used in this filter

        for i in range(3):
            # Q = np.array([[(dt**5)/20., ((dt**4)/8.)*80., (dt**3)/6.],
            #               [((dt**4)/8.)*80., ((dt**3)/8.), (dt**2)/2.],
            #               [(dt**3)/6., (dt**2)/2., dt]])
            # self.Q[3*i:3*i+3, 3*i:3*i+3] = Q*13.
            self.Q[3*i:3*i+3, 3*i:3*i+3] = Q_continuous_white_noise(3, dt, 13.)


        # accelerometer 3 axes
        # barometric altimeter 1 axis
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def priori(self, R: np.ndarray, T:float, m:float, r:float, Cn:float, Ca:float, rho:float, gyro:tuple):
        """
        Predicts the next state and covariance of the system using the Extended Kalman Filter (EKF) approach.
            R (np.ndarray): Rotation matrix representing the orientation of the body frame relative to the world frame.
            T (float): Thrust vector in the body frame.
            m (float): Mass of the current state of the rocket.
            r (float): Radius of the system (used for aerodynamic calculations).
            Cn (float): Normal force coefficient for aerodynamic forces.
            Ca (float): Axial force coefficient for aerodynamic forces.
            rho (float): Air density.
            gyro (tuple): Angular velocity of the system in the body frame (gyroscope readings).
            bno_attitude (tuple): Attitude of the system (e.g., roll, pitch, yaw) from the BNO sensor.
        Attributes Updated:
            x_priori (np.ndarray): Predicted state vector after applying the dynamics.
            P_priori (np.ndarray): Predicted covariance matrix after applying the dynamics.
            F (np.ndarray): Linearized dynamics matrix.

        Try reading this: https://en.wikipedia.org/wiki/Kalman_filter#Details
        Dynamics: https://www.intechopen.com/chapters/64567
        """
        # states tracked: x, vx, ax, y, vy, ay, z, vz, az

        g = 9.81 # Earth gravity
        vel_x, vel_y, vel_z = self.x_k[1], self.x_k[4], self.x_k[7]
        vel_mag = np.linalg.norm([vel_x, vel_y, vel_z])
        acc_x, acc_y, acc_z = self.x_k[2], self.x_k[5], self.x_k[8]

        vel_norm_inv = 0
        if (vel_y != 0 or vel_z != 0):
            vel_norm_inv = 1/np.sqrt(vel_y**2+vel_z**2)

        # TODO: grab angular velocity from r_EKF instead of gyro
        self.w_k = gyro
        w_x, w_y, w_z = self.w_k[0], self.w_k[1], self.w_k[2]

        # Calculate aerodynamic forces in body frame
        Fax = -0.5*rho*(vel_mag**2)*float(Ca)*(np.pi*r**2)
        Fay = 0.5*rho*(vel_mag**2)*(-Cn*vel_y*vel_norm_inv)*(np.pi*r**2)
        Faz = 0.5*rho*(vel_mag**2)*(Cn*vel_norm_inv)*(np.pi*r**2)

        # Transform forces to body frame
        Fg = np.array([-g, 0, 0])
        Fg_body = np.linalg.inv(R) @ Fg
        Fgx, Fgy, Fgz = Fg_body[0], Fg_body[1], Fg_body[2]
        Ftx, Fty, Ftz = T[0],T[1],T[2]

        
        xdot = np.array([vel_x, ((Fax + Ftx + Fgx) / m - (w_y*vel_z - w_z*vel_y) + acc_x) / 2, 0.0,
                 vel_y, ((Fay + Fty + Fgy) / m - (w_z*vel_x - w_x*vel_z) + acc_y) / 2, 0.0,
                 vel_z, ((Faz + Ftz + Fgz) / m - (w_x*vel_y - w_y*vel_x) + acc_z) / 2, 0.0
                ])
        
        self.x_priori = self.x_k + xdot * self.s_dt
        
        # linearized dynamics are F
        self.F = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1/2, 0, w_z/2, 0, 0, -w_y/2, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 1, 0, 0, 0, 0], 
                        [0, -w_z/2, 0, 0, 0, 1/2, 0, w_x/2, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 1, 0], 
                        [0, w_y/2, 0, 0, -w_x/2, 0, 0, 0, 1/2], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.P_priori = self.F @ self.P_k @ self.F.T + self.Q

    def update(self, bno_attitude, x_pos, x_accel, y_accel, z_accel):
        """Updates state and covariance
        
        Args:
            bno_attitude (tuple): (roll, pitch, yaw) in radians 
            x_pos (float): x position
            x_accel (float): x acceleration
            y_accel (float): y acceleration
            z_accel (float): z acceleration
        """
        
        K = (self.P_priori @ self.H.T) @ np.linalg.inv(self.H @ self.P_priori @ self.H.T + self.R)
        acc = vct.body_to_world(*bno_attitude, np.array([x_accel, y_accel, z_accel])) + np.array([-9.81, 0, 0])
        y_k = np.array([x_pos, *acc]).T

        self.x_k = self.x_priori + K @ (y_k - (self.H @ self.x_priori))
        self.P_k = (np.eye(len(K)) - K @ self.H) @ self.P_priori

        self.current_time += self.s_dt

    def get_state(self):
        """Returns current state
        
        Returns:
            np.array: current state
        """
        return self.x_k
    
    def get_covariance(self):
        """Returns current covariance
        
        Returns:
            np.array: current covariance
        """
        return np.diag(self.P_k)

    def reset_lateral_pos(self):
        """Resets lateral position to 0
        """
        self.x_k[1:3] = [0, 0]
        


