import numpy as np
import gym
from gym import spaces
import pandas as pd
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


class EnergyManagementEnv(gym.Env):
    def __init__(self, latitude, longitude, utility_prices, prior_purchased, num_time_intervals=480):
        super().__init__()
        self.num_time_intervals = num_time_intervals
        self.utility_prices = utility_prices
        self.prior_purchased = prior_purchased
        self.latitude = latitude
        self.longitude = longitude

        # Initialize location and solar power system
        self.location = Location(latitude, longitude)
        self.system = self.create_pv_system()
        self.mc = self.create_model_chain(self.system, self.location)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),
                                       high=np.array([1, self.battery_max_discharge, max(self.utility_prices)]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array([2000, self.battery_capacity, max(self.utility_prices)]),
                                            dtype=np.float32)

        self.load_profiles = self.generate_load_profiles()
        self.weather_data = self.generate_weather_data()

    def create_pv_system(self):
        module_parameters = {
            'pdc0': 240,  # Max power under STC in watts
            'gamma_pdc': -0.0045  # Temperature coefficient
        }
        inverter_parameters = {
            'Paco': 230,  # AC output capacity
            'Pdco': 240,  # Input DC power at rated AC output
            'Vdco': 48,  # DC voltage for rated AC output
            'Pso': 2.5,  # Power required to start the inverter
            'C0': -0.000041,
            'C1': -0.000091,
            'C2': 0.000494,
            'C3': -0.013171,
            'Pnt': 0.075  # Night tare loss
        }
        system = PVSystem(module_parameters=module_parameters, inverter_parameters=inverter_parameters,
                          temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])
        return system

    def create_model_chain(self, system, location):
        mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss',
                        dc_model='pvwatts', ac_model='sandia', temperature_model='sapm')
        return mc

    def generate_weather_data(self):
        times = pd.date_range(start='2022-01-01', periods=self.num_time_intervals, freq='3T', tz=self.location.tz)
        # Simulating clear sky data for simplicity
        weather = self.location.get_clearsky(times)  # Typical clear sky data
        weather['temp_air'] = 20  # 20 degrees Celsius
        weather['wind_speed'] = 5  # 5 m/s
        return weather

    def reset(self):
        self.current_interval = 0
        self.battery_state = self.battery_capacity / 2
        return np.array([self.load_profiles[0], self.battery_state, self.utility_prices[0]])

    def step(self, action):
        solar_usage, battery_discharge, utility_purchase = action
        weather = self.weather_data.iloc[
                  self.current_interval:self.current_interval + 1]  # Select the current weather slice
        self.mc.run_model(weather)
        solar_power = solar_usage * self.mc.results.ac  # AC power output from the solar panels

        total_power = solar_power + battery_discharge + utility_purchase + self.prior_purchased[self.current_interval]
        demand_met = min(self.load_profiles[self.current_interval], total_power)
        remaining_demand = max(0, self.load_profiles[self.current_interval])
