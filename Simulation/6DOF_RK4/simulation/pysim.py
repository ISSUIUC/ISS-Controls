# ______      _____ _              __________ ___________ 
# | ___ \    /  ___(_)            / ___|  _  \  _  |  ___|
# | |_/ /   _\ `--. _ _ __ ___   / /___| | | | | | | |_   
# |  __/ | | |`--. \ | '_ ` _ \  | ___ \ | | | | | |  _|  
# | |  | |_| /\__/ / | | | | | | | \_/ | |/ /\ \_/ / |    
# \_|   \__, \____/|_|_| |_| |_| \_____/___/  \___/\_|    
#        __/ |                                            
#       |___/                                             

# A 6DOF RK-4 Based simulation that uses RASAero aerodynamic data and known motor thrust data to simulate motion of the rocket
# with simulated sensor data as well as a implementation of the Extended Kalman Filter and active drag PID controller for the ISS
# Spaceshot entry for the 2023 IREC competition. This simulation was used to quantify the effects of the airbrakes, test 
# different system design methodlogies, and provide preliminary tuning for the EKF and controller prior to implementation in 
# SILSIM and flight software.
# 
# 2022-2023 Guidance, Navigation, and Control Main Contributors #
# Sub-Team Lead: Parth Shrotri (2024)
# Colin Kinsey (2024)
# Evan Yu (2025)
# Rithvik Bhogavilli (2025)
# Kabir Cheema (2025)
# Freya Bansal (2025)
# Ishaan Bansal (2025)
# Ethan Pereira (2026)

# 2024-2025 Guidance, Navigation, and Control Main Contributors #
# Sub-Team Lead: Shishir Bhatta (2026)
# Ishaan Kandamuri (2026)
# Aneesh Ganti (2028)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import estimation.ekf as ekf
import estimation.r_ekf as r_ekf
import properties.properties as prop
import properties.data_loader as dataloader
import simulator as sim_class
import dynamics.sensors as sensors
import time
import dynamics.rocket as rocket_model
import environment.atmosphere as atmosphere

# Load desired config file
config = dataloader.config
angleCheck = input("Nominal or Tilted Flight: ").lower()
tiltCommand = "Tilt".lower()

# Runs simulation for the specific component of the rocket
class Simulation:
    # dt can be dynamic in the future, so we need to 
    def __init__(self, rocket, motor, dt, x0, time_stamp=0, stages=[]):
        self.rocket = rocket
        self.stages = stages
        self.motor = motor
        self.dt = dt
        self.x = x0.copy()
        self.baro = 0
        self.time_stamp = time_stamp
        self.nominal = angleCheck
        if (self.nominal  == tiltCommand):
            sim.Vmultiplier = 30
        self.sensor_config = self.rocket.stage_config['sensors']

    # Call on Navigation class?

    def time_step(self):
        self.time_stamp += self.dt

    def idle_stage(self):
        while self.time_stamp < self.rocket.delay:
            baro_alt, accel, gyro, bno_ang_pos = self.get_sensor_data()
            self.rocket.Navigation.update_state(baro_alt, accel, gyro, bno_ang_pos, self.time_stamp)
            self.rocket.Navigation.kalman_filter.reset_lateral_pos()
            current_state, current_covariance, current_state_r = self.get_kalman_state()

            self.rocket.add_to_dict(self.x, baro_alt, accel, bno_ang_pos, gyro, current_state, current_covariance, current_state_r, 0, self.rocket.get_rocket_dry_mass(), self.rocket.get_total_motor_mass(self.time_stamp), 0, dt)
            self.time_step()

    def execute_stage(self):
        # Run the stages
        
        stage_separation_delay = 1
        self.rocket.get_motor().ignite(self.time_stamp)
        if(rocket.current_stage == -1):
            print(f"Take off at {self.time_stamp}")
        ignition_time = self.time_stamp
        start = True
        print(f"Staged at {self.time_stamp}")
        while self.time_stamp < ignition_time + self.rocket.get_motor().get_burn_time() + stage_separation_delay:
            # Get sensor data
            baro_alt, accel, gyro, bno_ang_pos = self.get_sensor_data()
            
            
            self.rocket.Navigation.update_state(baro_alt, accel, gyro, bno_ang_pos, self.time_stamp)
            current_state, current_covariance, current_state_r = self.get_kalman_state()

            self.rocket.set_motor_mass(self.time_stamp)

            is_staging = start and self.rocket.current_stage != -1
            self.x, alpha = sim.RK4(self.x, dt, self.time_stamp, is_staging, 0)

            self.rocket.add_to_dict(self.x, baro_alt, accel, bno_ang_pos, gyro, current_state, current_covariance, current_state_r, alpha, self.rocket.get_rocket_dry_mass(), self.rocket.get_total_motor_mass(self.time_stamp), 0, dt)
            self.time_step()
            if start:
                start = False

    def run_stages(self):
        has_more_stages = True
        while has_more_stages: 
            if(self.motor.burnout(self.time_stamp)):
                print(f"Motor burnout at {self.time_stamp}")
            self.execute_stage()
            has_more_stages = self.rocket.separate_stage(self.time_stamp) 

    # Function to retrive all sensor data
    def get_sensor_data(self):
        return (self.rocket.get_barometer_data(self.x, self.sensor_config),
                self.rocket.get_accelerometer_data(self.x, self.sensor_config),
                self.rocket.get_gyro_data(self.x, self.sensor_config), 
                self.rocket.get_bno_orientation(self.x, self.sensor_config))

    def get_kalman_state(self):
        current_state = self.rocket.Navigation.kalman_filter.get_state()
        current_cov = self.rocket.Navigation.kalman_filter.get_covariance()
        current_state_r = self.rocket.Navigation.r_kalman_filter.get_state()
        return (current_state, current_cov, current_state_r)

    def coast(self):
        if(self.motor.burnout(self.time_stamp)): 
            print(f"Motor burnout at {self.time_stamp}")
        while self.x[1, 0] >= 0:
        # Get sensor data
            baro_alt, accel, gyro, bno_ang_pos = self.get_sensor_data()
            self.rocket.Navigation.update_state(baro_alt, accel, gyro, bno_ang_pos, self.time_stamp)
            current_state, current_covariance, current_state_r = self.get_kalman_state()

            self.x, alpha = sim.RK4(self.x, dt, self.time_stamp, 0)

            self.rocket.add_to_dict(self.x, baro_alt, accel, bno_ang_pos, gyro, current_state, current_covariance, current_state_r, alpha, self.rocket.get_rocket_dry_mass(), self.rocket.get_total_motor_mass(self.time_stamp), 0, dt)
            self.time_step()
    
def simulator(x0, rocket, motor, dt) -> None:
    '''Method which handles running the simulation and logging sim data to dict

    Args:
        x0 (np.array): state vector initialized to 0s [6x3]
             x:         y:         z:
           [[pos,       pos,       pos],
            [vel,       vel,       vel],
            [accel,     accel,     accel],
            [ang_pos,   ang_pos,   ang_pos],
            [ang_vel,   ang_vel,   ang_vel],
            [ang_accel, ang_accel, ang_accel]]
        dt (float): time step between each iteration in simulation
    '''
    simulator = Simulation(rocket, motor, dt, x0, stages=rocket.stages)
    simulator.idle_stage()
    t_start = time.time()
    simulator.run_stages()
    simulator.coast()
    t_end = time.time() - t_start
    print(f"Runtime: {t_end:.2f} seconds")


if __name__ == '__main__':
    x0 = np.zeros((6, 3))
    x0[3] = [0, 0.05, 0]
    dt = 0.01

    atm = atmosphere.Atmosphere(enable_direction_variance=True, enable_magnitude_variance=True)

    stages = []
    for stage in config['rocket']['stages'][1:]:
        stages.append(rocket_model.Rocket(dt, x0, stage, atm=atm))
    rocket = rocket_model.Rocket(dt, x0, config['rocket']['stages'][0], atm=atm, stages=stages)

    motor = rocket.motor
    sim = sim_class.Simulator(atm=atm, rocket=rocket)

    simulator(x0, rocket, motor, dt)

    print("Writing to file...")

    record = rocket.to_csv()
   
    motorCutoffCount = True
    while motorCutoffCount:
         for point in range(len(rocket.sim_dict["time"])):
            if (rocket.sim_dict["alpha"][point] > 23):
                print("Motor cutoff should happen at time: ", rocket.sim_dict["time"][point])
                motorCutoffCount = False
                break
         motorCutoffCount = False

    # output_dir = os.path.join(os.path.dirname(__file__), config["meta"]["output_file"])
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir) 
    # output_file = os.path.join(output_dir, 'simulated_6dof.csv')
    # with open(output_file, 'w') as f:
    #     f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,accel_x,accel_y,accel_z,ang_pos_x,ang_pos_y,ang_pos_z,ang_vel_x,ang_vel_y,ang_vel_z,ang_accel_x,ang_accel_y,ang_accel_z,alpha,rocket_total_mass,motor_mass,flap_ext,baro_alt,imu_accel_x,imu_accel_y,imu_accel_z,imu_ang_pos_x,imu_ang_pos_y,imu_ang_pos_z,imu_gyro_x,imu_gyro_y,imu_gyro_z,kalman_pos_x,kalman_vel_x,kalman_accel_x,kalman_pos_y,kalman_vel_y,kalman_accel_y,kalman_pos_z,kalman_vel_z,kalman_accel_z,pos_cov_x,vel_cov_x,accel_cov_x,pos_cov_y,vel_cov_y,accel_cov_y,pos_cov_z,vel_cov_z,accel_cov_z,kalman_rpos_x,kalman_rvel_x,kalman_raccel_x,kalman_rpos_y,kalman_rvel_y,kalman_raccel_y,kalman_rpos_z,kalman_rvel_z,kalman_raccel_z\n")
    #     for point in record:
    #         f.write(f"{','.join(point)}\n")
    output_file = os.path.join(os.path.dirname(__file__), config["meta"]["output_file"])
    with open(output_file, 'w') as f:
        f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,accel_x,accel_y,accel_z,ang_pos_x,ang_pos_y,ang_pos_z,ang_vel_x,ang_vel_y,ang_vel_z,ang_accel_x,ang_accel_y,ang_accel_z,alpha,rocket_total_mass,motor_mass,flap_ext,baro_alt,imu_accel_x,imu_accel_y,imu_accel_z,imu_ang_pos_x,imu_ang_pos_y,imu_ang_pos_z,imu_gyro_x,imu_gyro_y,imu_gyro_z,kalman_pos_x,kalman_vel_x,kalman_accel_x,kalman_pos_y,kalman_vel_y,kalman_accel_y,kalman_pos_z,kalman_vel_z,kalman_accel_z,pos_cov_x,vel_cov_x,accel_cov_x,pos_cov_y,vel_cov_y,accel_cov_y,pos_cov_z,vel_cov_z,accel_cov_z,kalman_rpos_x,kalman_rvel_x,kalman_raccel_x,kalman_rpos_y,kalman_rvel_y,kalman_raccel_y,kalman_rpos_z,kalman_rvel_z,kalman_raccel_z\n")
        for point in record:
            f.write(f"{','.join(point)}\n")

