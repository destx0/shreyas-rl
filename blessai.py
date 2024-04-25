from env.energy_management import EnergyManagementEnv
import numpy as np
from methods.reinforcement_learning import run_rl
from methods.simulated_annealing import run_sa
from methods.optimization import run_optimization
from utils.plot_results import plot_results
from utils.tabulate_results import tabulate_results

def generate_rates(num_intervals):
    base_rate = 0.10
    peak_rate = 0.20
    utility_rates = np.random.uniform(base_rate, peak_rate, num_intervals)

    for i in range(num_intervals):
        hour = (i // (num_intervals // 24)) % 24
        if 8 <= hour <= 20:
            utility_rates[i] *= 1.5

    prior_purchased_rates = np.random.uniform(0.08, 0.12, num_intervals)  # More stable rates

    return utility_rates, prior_purchased_rates


utility_rates,prior_rates = generate_rates(480)

env = EnergyManagementEnv(latitude=80, longitude=72, utility_prices=utility_rates, prior_purchase=prior_rates)

rl_results = run_rl(env)
#sa_results = run_sa(env)
#opt_results = run_optimization(env)


#plot_results(rl_results, sa_results, opt_results)
#tabulate_results(rl_results, sa_results, opt_results)
