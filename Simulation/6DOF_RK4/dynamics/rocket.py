import math
import numpy as np
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import environment.atmosphere as atmosphere
import dynamics.motor as motor
from dynamics.motor import Motor
import dynamics.forces as forces
import util.vectors as vct
import random
import properties.properties as prop
import properties.data_loader as dataloader
import environment.atmosphere as atm
from estimation.navigation import Navigation

class Rocket:
    motor = None
    forces = None
    stage_config = None
    multiplier = 1

    def init_dicts(self):

        self.sim_dict = {
            "pos": [],
            "vel": [],
            "accel": [],
            "ang_pos": [],
            "ang_vel": [],
            "ang_accel": [],
            "alpha": [],
            "flap_ext": [],
            "rocket_total_mass": [],
            "motor_mass": [],
            "time": []
        }

        self.kalman_dict = {
            "x": [],
            "y": [],
            "z": [],
            "cov_x": [],
            "cov_y": [],
            "cov_z": [],
            "rx": [],
            "ry": [],
            "rz": [],
            "time": []
        }

        self.sensor_dict = {
            "baro_alt": [],
            "imu_accel_x": [],
            "imu_accel_y": [],
            "imu_accel_z": [],
            "imu_ang_pos_x": [],
            "imu_ang_pos_y": [],
            "imu_ang_pos_z": [],
            "imu_gyro_x": [],
            "imu_gyro_y": [],
            "imu_gyro_z": []
        }
        
        #Import dataframe from CSV so it doesn't have to call it every time

        dir = os.path.dirname(os.path.dirname(os. path.abspath(__file__)))
        csv_path = os.path.join(dir, "LookUp", "ekf_cd_test.csv")
        self.coeffs_df = pd.read_csv(csv_path)

        self.coeffs_dict = {
            "CN": [0],
            "CA Power-On": [0],
            "CA Power-Off": [0],
            "CD Power-On": [0],
            "CD Power-Off": [0],
            "CL": [0],
            "CP": [0]
        }


    def __init__(self, dt, x0, stage_config, atm:atmosphere.Atmosphere=None, stages:list=[]):
        self.stage_config = stage_config
        self.cm_rocket = stage_config["rocket_body"]["structure_cm"]
        self.cm_motor = stage_config["motor"]["cm"]
        self.cm = stage_config['rocket_body']['combined_cm']
        self.cp = stage_config['rocket_body']['combined_cp']
        
        self.dt = dt

        self.impulse = stage_config["motor"]["impulse"]
        self.motor_mass = stage_config["motor"]["motor_mass"]
        self.delay = stage_config["motor"]["delay"]
        self.motor_lookup_file = stage_config["motor"]["motor_lookup_file"]
        self.Navigation = Navigation(dt, stage_config['sensors'], x0, self) 

        self.rocket_dry_mass = stage_config["rocket_body"]["dry_mass"]
        self.rocket_total_mass = self.rocket_dry_mass + self.motor_mass
        self.r_r = stage_config["rocket_body"]["radius"]
        self.l = stage_config["rocket_body"]["length"]
        self.A = math.pi * self.r_r ** 2 
        self.A_s = 2 * self.r_r * self.l # surface area
        self.max_ext_length = stage_config["flaps"]["max_ext_length"]
        self.atm = atm
        self.init_dicts()
        
        # Initializing the coefficients
        # self.update_coeffs(0, 0)

        # Add stages to rocket via this list. Only the base rocket object should have stages, each stage should be its own rocket object with no stages
        self.stages = stages
        
        '''
        Start with the base rocket object, which is the first stage, then increment at each separation event.
        This is used to determine which stage the rocket is currently in. Start at -1, then increment to 0, 1, etc., so that
        they align with the indices of the stages list
        '''
        self.current_stage = -1
        
        '''
        This is used to store the time of the latest separation event. Start at 0, then update at each separation event so that timestamp is
        based on the latest separation event
        '''
        self.separation_timestamp = 0
        
        # need to change motor and forces constructors to refer to properties from this class rather than properties file
        self.motor = motor.Motor(self.rocket_dry_mass, 
                                 self.cm, 
                                 self.cm_rocket, 
                                 self.cm_motor, 
                                 self.rocket_dry_mass, 
                                 impulse=self.impulse, 
                                 mass=self.motor_mass, 
                                 delay=self.delay, 
                                 lookup_file=self.motor_lookup_file)
        self.forces = forces.Forces(self.max_ext_length,
                                    self.cm,
                                    self.cp,
                                    self.A,
                                    self.A_s,
                                    self.rocket_dry_mass,
                                    self.motor,
                                    stage_config["rocket_body"]["rasaero_lookup_file"],
                                    self.atm)
        #Do we need to pass self.coeffs_df into Forces?


    def get_total_motor_mass(self, timestamp) -> float:
        mass = 0
        if self.current_stage == -1:
            # Then get the mass of the motor
            mass += self.motor.get_mass(timestamp - self.separation_timestamp)
            # Sum all the other masses
            for x in self.stages:
                mass += x.get_motor().total_mass
            return mass

        mass += self.stages[self.current_stage].get_motor().get_mass(timestamp - self.separation_timestamp)
        for x in self.stages[self.current_stage+1:]:
            mass += x.get_motor().total_mass
        return mass

    def set_motor_mass(self, timestamp) -> None:
        """Sets the mass of the current stage's motor
        
        Args:
            timestamp (float): Time in seconds
        """
        if self.current_stage == -1:
            self.motor_mass = self.motor.get_mass(timestamp)
        else:
            self.motor_mass = self.stages[self.current_stage].get_motor().get_mass(timestamp - self.separation_timestamp)
        self.rocket_total_mass = self.rocket_dry_mass + self.get_total_motor_mass(timestamp)

    def is_motor_burnout(self, timestamp):
        return self.motor.burnout(timestamp)

    def get_motor_mass(self, timestamp) -> float:
        """Returns the mass of the motor at a given time
            Only used for upper stages, don't use for base stage
            use set_motor_mass for base stage
        
        Args:
            timestamp (float): Time in seconds
        
        Returns:
            float: Mass of the motor
        """
        return self.motor.get_mass(timestamp - self.separation_timestamp)
    
    def separate_stage(self, timestamp) -> bool:
        """ "Separates" the current stage from the rocket (increments the stage counter)
            and updates the separation timestamp
            
            Retuns True if there are more stages to separate, False if there are no more stages to separate
        """
        if self.current_stage == len(self.stages) - 1:
            return False
        self.current_stage += 1
        self.separation_timestamp = timestamp
        self.forces = self.stages[self.current_stage].forces
        return True

    def get_motor(self) -> Motor:
        """Returns the motor of the rocket
        """
        return self.motor if self.current_stage == -1 else self.stages[self.current_stage].get_motor()

    def get_rocket_dry_mass(self) -> float:
        """Returns the dry mass of the rocket
        """
        return self.rocket_dry_mass if self.current_stage == -1 else self.stages[self.current_stage].get_rocket_dry_mass()
    
    def get_rocket_total_mass(self, timestamp:float) -> float:
        return self.get_rocket_dry_mass() + self.get_total_motor_mass(timestamp)
 
    def get_CM(self):
        return self.cm if self.current_stage == -1 else self.stages[self.current_stage].get_CM()
    
    def get_CP(self):
        return self.cp if self.current_stage == -1 else self.stages[self.current_stage].get_CP()
    
    def get_A(self):
        return self.A if self.current_stage == -1 else self.stages[self.current_stage].get_A()
    
    def get_A_s(self):
        return self.A_s if self.current_stage == -1 else self.stages[self.current_stage].get_A_s()
    
    def get_Rasaero(self):
        return self.rasaero if self.current_stage == -1 else self.stages[self.current_stage].get_Rasaero()
    
    def get_cn(self):
        return self.coeffs_dict["CN"][-1]
    
    def get_ca_on(self):
        return self.coeffs_dict["CA Power-On"][-1]
    
    def get_ca_off(self):
        return self.coeffs_dict["CA Power-Off"][-1]
    
    def get_cd_on(self):
        return self.coeffs_dict["CD Power-On"][-1]
    
    def get_cd_off(self):
        return self.coeffs_dict["CD Power-Off"][-1]
    
    def get_cp(self):
        return self.coeffs_dict["CP"][-1]

    def I(self, total_mass): 
        """Returns the inertia matrix of the rocket
        
        Args:
            total_mass (float, optional): Total mass of the rocket. Defaults to rocket_total_mass.
        
        Returns:
            np.array: Inertia matrix of the rocket
        """
        return np.diag([(1/2) * total_mass * self.r_r**2,
                                       (total_mass/12) * (self.l**2 + 3*self.r_r**2),
                                       (total_mass/12) * (self.l**2 + 3*self.r_r**2)])

    

    def I_inv(self, total_mass) -> np.ndarray: 
        """Returns the inverse of the inertia matrix of the rocket
        
        Args:
            total_mass (float, optional): Total mass of the rocket. Defaults to rocket_total_mass.
        """
        return np.diag([1/((1/2) * total_mass * self.r_r**2),
                                       1/((total_mass/12) * (self.l**2 + 3*self.r_r**2)),
                                       1/((total_mass/12) * (self.l**2 + 3*self.r_r**2))])
    
    def get_accelerometer_data(self, x_state, sensor_config):
        """Returns the accelerometer data in the body frame of the rocket.
    
        Args:
            x_state: The state of the rocket in the world frame.
    
        Returns:
            accel_reading (np.ndarray): The accelerometer data in the body frame of the rocket. (Specific force)
        """
        gravity = [(prop.G*prop.m_e)/((prop.r_e+x_state[0,0])**2), 0, 0]
        body_accel = vct.world_to_body(*x_state[3], x_state[2] + gravity)
        kx134_rms = sensor_config["high_g"]["RMS"] * 9.81/1000 # Convert to m/(s^2)
        accel_reading = np.array([random.gauss(body_accel[0], kx134_rms), 
                      random.gauss(body_accel[1], kx134_rms), 
                      random.gauss(body_accel[2], kx134_rms)])
        return accel_reading
    
    def get_gyro_data(self, x_state, sensor_config):
        """Returns the gyroscope data in the body frame of the rocket.
    
        Args:
            x_state: The state of the rocket in the world frame.
        
        Returns:
            gyro_reading (np.ndarray): The gyroscope data in the body frame of the rocket. (Angular velocity)
        """
        body_accel = vct.world_to_body(*x_state[3], x_state[4])
        gyro_rms = sensor_config["gyro"]["RMS"] * np.pi / 180000
        gyro_reading = np.array([random.gauss(body_accel[0], gyro_rms), 
                             random.gauss(body_accel[1], gyro_rms), 
                             random.gauss(body_accel[2], gyro_rms)])
        return gyro_reading

    def get_barometer_data(self, x_state, sensor_config):
        """Returns the barometer data in the world frame of the rocket.
    
        Args:
            x_state: The state of the rocket in the world frame.
    
        Returns:
            altitude (float): The altitude of the rocket in meters. (Barometric)
        """
        a = atm.Atmosphere()
        baro_rms = sensor_config["barometer"]["RMS"] * 100 # Pascals
    
        pressure = a.get_pressure(x_state[0, 0])
        pressure = random.gauss(pressure, baro_rms)
        altitude = a.get_altitude(pressure)
    
        return altitude
    
    def get_bno_orientation(self, x_state, sensor_config):
        """Returns the sensor emulated orientation data in the world frame of the rocket.
    
        Args:
            x_state: The state of the rocket in the world frame.
        
        Returns:
            bno_reading (np.ndarray): The sensor emulated orientation data in the world frame of the rocket. (Euler angles)
        """
        true_orientation = x_state[3]
        bno_error = sensor_config["bno"]["error"] * np.pi / 180
        bno_reading = np.array([random.gauss(true_orientation[0], bno_error), 
                             random.gauss(true_orientation[1], bno_error), 
                             random.gauss(true_orientation[2], bno_error)])
        return bno_reading
    
    
    # Appends data to rocket's storage
    def add_to_dict(self, x, baro_alt, accel, bno_ang_pos, gyro, kalman_filter, kf_cov, kalman_filter_r, alpha, rocket_total_mass, motor_mass, flap_ext, dt):
        # Append to sensor_dict
        self.sensor_dict["baro_alt"].append(baro_alt)
        self.sensor_dict["imu_accel_x"].append(accel[0])
        self.sensor_dict["imu_accel_y"].append(accel[1])
        self.sensor_dict["imu_accel_z"].append(accel[2])
        self.sensor_dict["imu_ang_pos_x"].append(bno_ang_pos[0])
        self.sensor_dict["imu_ang_pos_y"].append(bno_ang_pos[1])
        self.sensor_dict["imu_ang_pos_z"].append(bno_ang_pos[2])
        self.sensor_dict["imu_gyro_x"].append(gyro[0])
        self.sensor_dict["imu_gyro_y"].append(gyro[1])
        self.sensor_dict["imu_gyro_z"].append(gyro[2])
        

        self.kalman_dict["x"].append(kalman_filter[0:3])
        self.kalman_dict["y"].append(kalman_filter[3:6])
        self.kalman_dict["z"].append(kalman_filter[6:9])
        self.kalman_dict["cov_x"].append(kf_cov[0:3])
        self.kalman_dict["cov_y"].append(kf_cov[3:6])
        self.kalman_dict["cov_z"].append(kf_cov[6:9])
        self.kalman_dict["rx"].append(kalman_filter_r[0:3])
        self.kalman_dict["ry"].append(kalman_filter_r[3:6])
        self.kalman_dict["rz"].append(kalman_filter_r[6:9])

        # Update Simulator Log
        self.sim_dict["pos"].append(x[0])
        self.sim_dict["vel"].append(x[1])
        self.sim_dict["accel"].append(x[2])
        self.sim_dict["ang_pos"].append(x[3])
        self.sim_dict["ang_vel"].append(x[4])
        self.sim_dict["ang_accel"].append(x[5])
        self.sim_dict["time"].append(self.sim_dict["time"][-1] + dt if len(self.sim_dict["time"]) > 0 else 0)
        self.sim_dict["flap_ext"].append(flap_ext)                    
        self.sim_dict["alpha"].append(alpha)
        self.sim_dict["rocket_total_mass"].append(rocket_total_mass)
        self.sim_dict["motor_mass"].append(motor_mass)

        #Update coefficients (how to find angle of attack???)
        self.update_coeffs(np.linalg.norm(x[1]))

    def update_coeffs(self, velocity):
        a = velocity / 340.29
        if (a < 0.01):
            a = 0.01
        df_specific = self.coeffs_df[(self.coeffs_df["Alpha"] == 2) & (self.coeffs_df["Mach"] == round(a, 2))]
        self.coeffs_dict["CN"].append(df_specific["CN"].values[0])
        self.coeffs_dict["CA Power-On"].append(df_specific["CA Power-On"].values[0])
        self.coeffs_dict["CA Power-Off"].append(df_specific["CA Power-Off"])
        self.coeffs_dict["CD Power-On"].append(df_specific["CD Power-On"])
        self.coeffs_dict["CD Power-Off"].append(df_specific["CD Power-Off"])
        self.coeffs_dict["CL"].append(df_specific["CL"])
        self.coeffs_dict["CP"].append(df_specific["CP"])

    # Converts the data saved in this sim into csv
    def to_csv(self):
        #Output
        record = []
        for point in range(len(self.sim_dict["time"])):
            cur_point = []
            cur_point.append(str(self.sim_dict["time"][point]))
            cur_point += list(map(str, self.sim_dict["pos"][point]))
            cur_point += list(map(str, self.sim_dict["vel"][point]))
            cur_point += list(map(str, self.sim_dict["accel"][point]))
            cur_point += list(map(str, self.sim_dict["ang_pos"][point]))
            cur_point += list(map(str, self.sim_dict["ang_vel"][point]))
            cur_point += list(map(str, self.sim_dict["ang_accel"][point]))
            cur_point += map(str, list([self.sim_dict["alpha"][point]]))
            cur_point += map(str, list([self.sim_dict["rocket_total_mass"][point]]))
            cur_point += map(str, list([self.sim_dict["motor_mass"][point]]))
            cur_point += map(str, list([self.sim_dict["flap_ext"][point]]))
            cur_point += map(str, list([self.sensor_dict["baro_alt"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_accel_x"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_accel_y"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_accel_z"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_ang_pos_x"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_ang_pos_y"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_ang_pos_z"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_gyro_x"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_gyro_y"][point]]))
            cur_point += map(str, list([self.sensor_dict["imu_gyro_z"][point]]))
            cur_point += map(str, list(self.kalman_dict["x"][point]))
            cur_point += map(str, list(self.kalman_dict["y"][point]))
            cur_point += map(str, list(self.kalman_dict["z"][point]))
            cur_point += map(str, list(self.kalman_dict["cov_x"][point]))
            cur_point += map(str, list(self.kalman_dict["cov_y"][point]))
            cur_point += map(str, list(self.kalman_dict["cov_z"][point]))
            cur_point += map(str, list(self.kalman_dict["rx"][point]))
            cur_point += map(str, list(self.kalman_dict["ry"][point]))
            cur_point += map(str, list(self.kalman_dict["rz"][point]))

            record.append(cur_point)
        return record
