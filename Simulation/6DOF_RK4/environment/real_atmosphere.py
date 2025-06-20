import requests
import datetime
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv('local.env')

class RealAtmosphere:
    def __init__(self, latitude, longitude, date=None, time=None, altitude=None, google_maps_api_key=None):
        self.latitude = latitude
        self.longitude = longitude
        self.date = date or datetime.date.today()
        self.time = time
        if google_maps_api_key is None:
            google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.google_maps_api_key = google_maps_api_key
        self.altitude = altitude if altitude is not None else self._get_altitude()
        self.weather_data = None
        self._validate_date()
        self._fetch_weather()
        self.nominal_wind_direction_ = None
        self.nominal_wind_magnitude_ = None
        self.enable_direction_variance_ = False
        self.enable_magnitude_variance_ = False

    def _get_altitude(self):
        if not self.google_maps_api_key:
            print("No Google Maps API key provided, using default altitude 0m.")
            return 0
        url = (
            f'https://maps.googleapis.com/maps/api/elevation/json?locations={self.latitude},{self.longitude}'
            f'&key={self.google_maps_api_key}'
        )
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data['status'] == 'OK' and data['results']:
                    return data['results'][0]['elevation']
                else:
                    print(f"Google Elevation API error: {data.get('status', 'Unknown error')}. Using 0m.")
            else:
                print(f"Failed to fetch elevation: {resp.text}. Using 0m.")
        except Exception as e:
            print(f"Exception during elevation lookup: {e}. Using 0m.")
        return 0

    def _validate_date(self):
        today = datetime.date.today()
        if not (today - datetime.timedelta(days=7) <= self.date <= today + datetime.timedelta(days=7)):
            raise ValueError("Date must be today or within 7 days from today.")

    def _pst_to_utc_datetime(self):
        if self.time is None:
            hour, minute = 12, 0
        else:
            hour = self.time // 100
            minute = self.time % 100
        dt_pst = datetime.datetime.combine(self.date, datetime.time(hour, minute))
        dt_utc = dt_pst + datetime.timedelta(hours=8)
        return dt_utc.replace(tzinfo=datetime.timezone.utc)

    def _fetch_weather(self):
        headers = {'User-Agent': 'Arthur (keshavbalaji2397@gmail.com)'}
        points_url = f'https://api.weather.gov/points/{self.latitude},{self.longitude}'
        resp = requests.get(points_url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch NWS grid info: {resp.text}")
        points_data = resp.json()
        forecast_hourly_url = points_data['properties']['forecastHourly']
        resp = requests.get(forecast_hourly_url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch NWS hourly forecast: {resp.text}")
        hourly_data = resp.json()
        periods = hourly_data['properties']['periods']
        target_dt = self._pst_to_utc_datetime()
        def parse_iso8601(s):
            return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
        closest = min(periods, key=lambda p: abs(parse_iso8601(p['startTime']) - target_dt))
        self.weather_data = closest

    def get_temperature(self, altitude=None):
        if self.weather_data:
            return self.weather_data['temperature'] + 273.15
        return None

    def get_pressure(self, altitude=None):
        if altitude is None:
            altitude = self.altitude
        P_0 = 101325

        # pressure under standard condition in (Pa)
        T_0 = 288.16 

        # Temperature lapse rate in k/m assuming temperature varies linearly based on altitude 
        b = 0.0065

        # gravitational constant 
        g = 9.81

        R = 287.05

        return P_0 * ((T_0 +(altitude)*b)/T_0)**(-g/(b*R))

    def get_density(self, altitude=None, noise=False, position=np.array([0,0,0])):
        R = 287.05
        p = self.get_pressure(altitude)
        T = self.get_temperature()
        if p is not None and T is not None:
            return p / (R * T)
        return None

    def get_speed_of_sound(self, altitude=None):
        gamma = 1.4
        gas_constant = 287.05
        T = self.get_temperature()
        if T is not None:
            return np.sqrt(gamma * gas_constant * T)
        return None

    def get_altitude(self, pressure):
        P_0 = 101325
        T_0 = 288.15
        b = 0.0065
        g = 9.81
        R = 287.05
        pressureRatio = pressure / P_0
        return -(T_0*((pressureRatio)**(b*R/(g)) - 1) * (pressureRatio)**(-b*R/(g)))/b

    def get_wind_vector(self, tStamp=None):
        speed, deg = self.get_wind()
        if speed is None or deg is None:
            return np.zeros(3)
        mph = float(speed.split()[0])
        mps = mph * 0.44704
        deg_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
                   'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
        deg_val = deg_map.get(deg.upper(), 0.0)
        rad = np.deg2rad(deg_val)
        wind_vec = np.array([
            mps * np.sin(rad),
            mps * np.cos(rad),
            0.0
        ])
        return wind_vec

    def get_wind(self):
        if self.weather_data:
            speed = self.weather_data.get('windSpeed', None)
            deg = self.weather_data.get('windDirection', None)
            return speed, deg
        return None, None

    def get_nominal_wind_direction(self):
        _, deg = self.get_wind()
        deg_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':135, 'SSE':157.5,
                   'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}
        deg_val = deg_map.get(deg.upper(), 0.0) if deg else 0.0
        rad = np.deg2rad(deg_val)
        return np.array([
            np.sin(rad),
            np.cos(rad),
            0.0
        ])

    def get_nominal_wind_magnitude(self):
        speed, _ = self.get_wind()
        if speed is None:
            return 0.0
        mph = float(speed.split()[0])
        return mph * 0.44704

    def set_nominal_wind_direction(self, direction):
        self.nominal_wind_direction_ = direction

    def set_nominal_wind_magnitude(self, magnitude):
        self.nominal_wind_magnitude_ = magnitude

    def toggle_wind_direction_variance(self, toggle):
        self.enable_direction_variance_ = toggle

    def toggle_wind_magnitude_variance(self, toggle):
        self.enable_magnitude_variance_ = toggle

    def summary(self):
        temp = self.get_temperature()
        pres = self.get_pressure()
        wind = self.get_wind()
        wind_vector = self.get_wind_vector()
        dens = self.get_density()
        return {
            'temperature_K': temp,
            'pressure_Pa': pres,
            'wind_speed': wind[0],
            'wind_direction': wind[1],
            'wind_vector': wind_vector,
            'density_kgm3': dens,
            'altitude_m': self.altitude
        }
