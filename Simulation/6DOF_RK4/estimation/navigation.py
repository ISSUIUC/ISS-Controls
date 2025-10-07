import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import estimation.ekf as ekf
import estimation.r_ekf as r_ekf
import dynamics.sensors as sensors
import environment.atmosphere as Atmosphere
import util.vectors as vct
## Class for the Rocket class and object to interact with the Kalman filter (and eventually staging optimization)
## Pysim and other files should not directly interact with Navigation or the ekf/r_ekf files but should instead utilize this class through a rocket object
class Navigation:
    def __init__(self, dt, sensor_config, x0, rocket):
        self.kalman_filter = ekf.KalmanFilter(dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.sensor_config = sensor_config
        self.x = x0
        self.rocket = rocket
        self.atm = Atmosphere.Atmosphere()
        
        x_data = np.empty(10)
        y_data = np.empty(10)
        z_data = np.empty(10)
        for i in range(10):
            reading = sensors.get_accelerometer_data(self.x, self.sensor_config)
            x_data[i] = reading[0]
            y_data[i] = reading[1]
            z_data[i] = reading[2]
        ## Acceleration value given by an average of a sum of values
        ax = sum(x_data)/len(x_data)
        ay = sum(y_data)/len(y_data)
        az = sum(z_data)/len(z_data)
        ## Angular component calculations
        #pitch = -1 * (np.arctan2(-az,-ay) + np.pi/2)
        #yaw = np.arctan2(-ax,-ay) + np.pi/2
        self.r_kalman_filter = r_ekf.KalmanFilter_R(dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        #self.r_kalman_filter = r_ekf.KalmanFilter_R(dt, 0.0, 0.0, 0.0, pitch, 0.0, 0.0, yaw, 0.0, 0.0)
        ## self.apogee_estimator = apg.Apogee(self.kalman_filter.get_state(), 0.1, 0.01, 3, 30, atm, self.rocket.stage_config)
        
    def update_state(self, baro_alt, accel, gyro, bno_ang_pos, timestep):
        
        roll, pitch, yaw = bno_ang_pos
        Rotational_matrix = vct.body_to_world(roll, pitch, yaw, np.eye(3))
        rho = self.atm.get_density(baro_alt)
        Cn = self.rocket.get_cn()
        Ca = self.rocket.get_ca_on()
        Cp = self.rocket.get_cp()
        Thrust = self.rocket.get_motor().get_thrust(timestep)
        m = self.rocket.get_rocket_total_mass(timestep)
        r = self.rocket.r_r
        h = self.rocket.l

        self.kalman_filter.priori(Rotational_matrix, Thrust, m, r, Cn, Ca, rho, gyro)
        self.r_kalman_filter.priori()
        self.kalman_filter.update(bno_ang_pos, baro_alt, accel[0], accel[1], accel[2])
        self.r_kalman_filter.update(*gyro, *accel)
        
        
