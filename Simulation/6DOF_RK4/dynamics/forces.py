'''
Forces on rocket:
- gravity
- motor thrust
- aerodynamic
    - wind (comes from atmosphere class)
'''
import dynamics.motor as motor
import environment.atmosphere as atmosphere
import properties.properties as prop
import numpy as np
import util.vectors as vct
import pandas as pd
import os
import random

# Define Objects

class Forces:
    """Forces on rocket:
    
    Args:
        max_ext_length (float): maximum extension length of flaps
        cm (np.array): center of mass of rocket
        cp (np.array): center of pressure of rocket
        A (float): cross sectional area of rocket
        A_s (float): cross sectional area of the side of the rocket
        rocket_dry_mass (float): mass of rocket without motor
        motor (motor.Motor): motor object
        atm (atmosphere.Atmosphere): atmosphere object
    """
    # Define class objects
    atm = None
    motor = None
    
    rasaero_file_location = "" # Will be set in constructor
    rasaero = None

    def __init__(self, max_ext_length, cm, cp, A, A_s, rocket_dry_mass, motor, rasaero_lookup_file, atm, multiplier=1):
        self.max_ext_length = max_ext_length
        self.cm = cm
        self.cp = cp
        self.A = A
        self.A_s = A_s
        self.rocket_dry_mass = rocket_dry_mass
        self.motor = motor
        self.atm = atm
        self.multiplier = multiplier
        self.rasaero_file_location = os.path.join(os.path.dirname(__file__), rasaero_lookup_file)
        self.rasaero = pd.read_csv(self.rasaero_file_location)

    

    def get_force(self, x_state, flap_ext, time_stamp, ejection_force, theta, phi, multiplier, density_noise=False) -> np.ndarray:
        '''Calculates net force felt by rocket while accounting for thrust, drag, gravity, wind

        Args:
            x_state (np.array): State Vector [4x3]
            flap_ext (float): current flap extention config
            time_stamp (float): current time stamp of rocket in simulation
        
        Returns:
            (np.array): 2D array of forces and moments --> ([Fx, Fy, Fz], [Mx, My, Mz])
            (float): angle of attack (radians)
        '''
        # TODO: Add random disturbances
        alt = x_state.copy()[0,0]
        density = self.atm.get_density(alt, noise=density_noise, position=x_state[0])
        thrust = self.motor.get_thrust(time_stamp)
        wind_vector = self.atm.get_wind_vector(time_stamp)
        alpha = self.get_alpha(x_state, wind_vector) 
        drag = self.aerodynamic_force(x_state, density, wind_vector, alpha, self.rasaero, thrust.dot(thrust) > 0, flap_ext)
        grav = self.gravitational_force(alt, time_stamp)
        force = vct.body_to_world(*x_state[2],thrust + drag) + grav
        moment = vct.body_to_world(*x_state[2], np.cross(-self.cm, thrust) + self.aerodynamic_moment(drag))
        if ejection_force != 0:
            dir = np.array([np.cos(phi), np.sin(phi)* np.sin(theta), np.sin(phi) * np.cos(theta)])
            force += ejection_force * dir
            moment += ejection_force * self.stages[self.current_stage].cm * dir
        return np.array([force, moment]), alpha

    def get_Ca_Cn_Cp(self, x_state, alpha, rasaero, before_burnout, flap_ext) -> list:
        '''References lookup table to find C_a, C_n, C_p based on flap extension

        Args:
            x_state (np.array): State Vector [4x3]
            alpha (float): angle between world x-axis and rocket x-axis (radians)
            rasaero (csv): lookup file for flight properties
            before_burnout (float): if greater than 0, then it is before burnout
            flap_ext (float): current flap extention config
                    
        Returns:
            [Ca, Cn, Cp]
            Ca (float): Coefficient of Axial Force
            Cn (float): Coefficient of Normal Force
            Cp (np.array): Coefficient of Pressure
        '''
        alt = x_state[0,0]
        vel = x_state[1]
        mach_number = np.linalg.norm(vel) / self.atm.get_speed_of_sound(alt)

        # Define mach number for csv lookup, rounded to hundreds place
        mach = round(mach_number, 2)

        #Define blank upper and lower Cas
        Ca_low = 0
        Ca_up = 0

        #Define blank upper and lower Cns
        Cn_low = 0
        Cn_up = 0

        # define csv file to search throughs
        csv_file = rasaero

        # Define protuberance percentage of full extension given current extension
        protub_perc = flap_ext/self.max_ext_length

        # Define starting Ca
        Ca = 0
        # Define starting Cn
        Cn = 0
        # Define starting Cp
        Cp = 0

        ca_vals = csv_file["CA Power-Off"]
        if (before_burnout):
            ca_vals = csv_file["CA Power-On"]
        
        cn_vals = csv_file["CN Total"]
        cp_vals = csv_file["CP Total"]

        # Find indices where the mach values match up
        mach_indices = np.where(csv_file['Mach Number'] == mach)[0]    
        
        if len(mach_indices) == 0:
            return [0,0,0]

        # Interpolate to find Cn value
        for idx in range(mach_indices[0], mach_indices[-1] + 1):
            if (round(np.degrees(alpha)) == csv_file['Alpha (deg)'][idx] and csv_file['Protuberance (%)'][idx] <= protub_perc <= csv_file['Protuberance (%)'][idx+1]):
                
                Ca_low = ca_vals[idx]
                Ca_up = ca_vals[idx+1]

                Cn_low = cn_vals[idx]
                Cn_up = cn_vals[idx+1]

                Cp_low = cp_vals[idx]
                Cp_up = cp_vals[idx+1]

                Ca = np.interp(protub_perc, [csv_file['Protuberance (%)'][idx], csv_file['Protuberance (%)'][idx+1]], [Ca_low, Ca_up])  
                Cn = np.interp(protub_perc, [csv_file['Protuberance (%)'][idx], csv_file['Protuberance (%)'][idx+1]], [Cn_low, Cn_up])  
                Cp = np.interp(protub_perc, [csv_file['Protuberance (%)'][idx], csv_file['Protuberance (%)'][idx+1]], [Cp_low, Cp_up])  
                Cp = Cp/100. #Convert cm to m

                return [Ca,Cn,np.array([Cp, 0.0, 0.0])]
            
        return [0,0,0]

    def aerodynamic_force(self, x_state, density, wind_vector, alpha, rasaero, before_burnout, flap_ext) -> np.ndarray:
        '''Calculates aerodynamic drag force acting on rocket based on velocity and altitude

        Args:
            x_state (np.array): State Vector [4x3]
            flap_ext (float): current flap extention config
        
        Returns:
            (np.array): vector of aerodynamic forces in each axis [1x3]
        '''
        vel = vct.world_to_body(*x_state[2].copy(), x_state[1].copy() - wind_vector.copy())
        C_a,C_n,self.cp = self.get_Ca_Cn_Cp(x_state, alpha, self.rasaero, before_burnout, flap_ext)
        roll_aero = np.arctan2(x_state[1,2], x_state[1,1])

        C_n_y = np.abs(C_n * np.cos(roll_aero)) #TODO: Check with other values
        C_n_z = np.abs(C_n * np.sin(roll_aero)) #TODO: Check with other values
        aero_force = -0.5*np.array([np.sign(vel[0])*vel[0]**2 * C_a*density*self.A, 
                                    np.sign(vel[1])*vel[1]**2 * C_n_y*density*self.A_s, 
                                    np.sign(vel[2])*vel[2]**2 * C_n_z*density*self.A_s])
        return aero_force
    
    def gravitational_force(self, altitude, time_stamp) -> np.ndarray:
        '''Calculates gravitational force acting on rocket based on altitude
        Relevant Equations:
            F = GMm/r^2

        Args:
            altitude (float): current altitude of rocket
            time_stamp (float): current time stamp of rocket in simulation
        
        Returns:
            (np.array): vector of gravitational forces on each axis [1x3]
        '''
        total_mass = self.rocket_dry_mass + self.motor.get_mass(time_stamp) #Adding dry mass + motor mass

        return -np.array([(prop.G*prop.m_e*total_mass)/((prop.r_e+altitude)**2), 0, 0])

    def aerodynamic_moment(self, aerodynamic_force) -> np.ndarray:
        aerodynamic_moment = np.cross(self.cp - self.motor.cm, aerodynamic_force)
        return aerodynamic_moment
        
    def get_alpha(self, x_state, wind_vector) -> float:
        incident_velocity = vct.world_to_body(*x_state[2], vct.norm(x_state[1] + wind_vector))
        orientation = vct.world_to_body(*x_state[2], np.array([1,0,0]))
        alpha = np.arccos(np.dot(incident_velocity, orientation))
        if(np.linalg.norm(incident_velocity) == 0):
            alpha = 0
        return alpha * self.multiplier
    
    