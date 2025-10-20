import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import util.vectors as vct
import dynamics.rocket as rocket_model
import environment.atmosphere as atm_model

class Simulator():
    
    atm = None
    rocket = None
    forces = None
    Vmultiplier = 1
    
    def __init__(self, atm: atm_model.Atmosphere, rocket: rocket_model.Rocket):
        self.atm = atm
        self.rocket = rocket
        
    ### TEST PURPOSES ###
    def newtonProp(self, y0, dt, time_stamp, flap_ext=0) -> np.ndarray:
        temp = (self.rocket.forces.get_force(np.array([y0[0], y0[1], y0[3], y0[4]]), flap_ext, time_stamp, self.Vmultiplier))
        a = temp[0]/self.rocket.rocket_total_mass

        moment = temp[1]
        alpha = moment
        v = y0[1] + a*dt
        p = y0[0] + v*dt + 0.5*a*dt**2
        return np.array([p, v, a, y0[3], y0[4], alpha])

    def RK4(self, y0, dt, time_stamp, flap_ext=0, staging=False, staging_noise= False, density_noise=False) -> np.ndarray:
        '''Propogates State Matrix of rocket based on Runge-Kutta (RK4) Method
        Args:
            y0 (np.array): current state vector [6x3]
                x:         y:         z:
            [[pos,       pos,       pos],
                [vel,       vel,       vel],
                [accel,     accel,     accel],
                [ang_pos,   ang_pos,   ang_pos],
                [ang_vel,   ang_vel,   ang_vel],
                [ang_accel, ang_accel, ang_accel]]
            dt (float): time step between each iteration in simulation
            time_stamp (float): current time stamp of rocket in simulation
            density_noise (bool): whether or not to add noise to atmospheric density
        
        Returns:
            (np.array): state vector of rocket in x-axis [6x3]
        '''
        Rk4Multiplier = self.Vmultiplier
        if staging:
            ejection_force = (random.gauss(0, 1) if staging_noise else 0)+12
            ejection_theta = random.gauss(0, .05) if staging_noise else 0
            ejection_phi = random.gauss(0, .05) if staging_noise else 0
        else:
            ejection_force = 0
            ejection_theta = 0
            ejection_phi = 0
        
        rocket_total_mass = self.rocket.get_rocket_total_mass(time_stamp)
        I_inv = self.rocket.I_inv(rocket_total_mass)
        k1_v = y0[2].copy()

        k2_v,k2_av = self.step_v(y0[0], y0[1] + (dt/2)*k1_v, y0[3], y0[4], time_stamp, ejection_force, ejection_theta, ejection_phi, flap_ext)
        k3_v,k3_av = self.step_v(y0[0], y0[1] + (dt/2)*k2_v, y0[3], y0[4], time_stamp, ejection_force, ejection_theta, ejection_phi, flap_ext)
        k4_v,k4_av = self.step_v(y0[0], y0[1] + dt*k3_v, y0[3], y0[4], time_stamp, ejection_force, ejection_theta, ejection_phi, flap_ext)

        k2_v /= rocket_total_mass
        k3_v /= rocket_total_mass
        k4_v /= rocket_total_mass

        v = (y0[1] + (1/6)*(k1_v+(2*k2_v)+(2*k3_v)+k4_v)*dt)

        k1_av = y0[5].copy()
        k2_av = I_inv@k2_av
        k3_av = I_inv@k3_av
        k4_av = I_inv@k4_av

        ang_v = (y0[4] + (1/6)*(k1_av+(2*k2_av)+(2*k3_av)+k4_av)*dt)

        k1_p = y0[1].copy()
        k2_p = self.step_p(y0[0], y0[0] + (dt/2)*k1_p, dt/2)
        k3_p = self.step_p(y0[0], y0[0] + (dt/2)*k2_p, dt/2)
        k4_p = self.step_p(y0[0], y0[0] + dt*k3_p, dt)

        p = (y0[0] + (1/6)*(k1_p+(2*k2_p)+(2*k3_p)+k4_p)*dt)

        k1_ap = y0[4].copy()
        k2_ap = self.step_p(y0[3], y0[3] + (dt/2)*k1_ap, dt/2)
        k3_ap = self.step_p(y0[3], y0[3] + (dt/2)*k2_ap, dt/2)
        k4_ap = self.step_p(y0[3], y0[3] + dt*k3_ap, dt)

        ang_p = (y0[3] + (1/6)*(k1_ap+(2*k2_ap)+(2*k3_ap)+k4_ap)*dt)

        temp,alpha = (self.rocket.forces.get_force(np.array([p, v, y0[3], y0[4]]), flap_ext, time_stamp, ejection_force, ejection_theta, ejection_phi, density_noise=density_noise, multiplier = self.Vmultiplier))
        alpha *= Rk4Multiplier
        a = temp[0]/rocket_total_mass

        return np.array([p, v, a, ang_p, ang_v, I_inv @ temp[1]]), alpha

    def step_p(self, y0, y1, dt):
        '''Calculates rate of change of position over given delta time for state propogation

        Args:
            y0 (np.array): current state vector [6x3]
            y1 (np.array): propogated state vector [6x3]
            dt (float): time step between iteration of RK4 (shorter than simulation dt)
        
        Returns:
            (np.array): rate of change of position (velocity) in form of state vector
        '''

        return (y1-y0)/dt # return slope (velocity)

    def step_v(self, pos, vel, ang_pos, ang_vel, time_stamp, ejection_force, ejection_theta, ejection_phi, flap_ext):
        '''Calculates slope of v over given delta t for state propogation

        Args:
            pos (np.array): current posiiton state vector [1x3]
            vel (np.array): current velocity state vector [1x3]
            dt (float): time step between iteration of RK4 (shorter than simulation dt)
            flap_ext (float): current flap extention config
            time_stamp (float): current time stamp of rocket in simulation
        
        Returns:
            (np.array): rate of change of velocity (acceleration) in form of state vector
        '''
        return self.rocket.forces.get_force(np.array([pos, vel, ang_pos, ang_vel]), flap_ext, time_stamp, ejection_force, ejection_theta, ejection_phi, multiplier = self.Vmultiplier)[0] # return slope times mass/inertia 
