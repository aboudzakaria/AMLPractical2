"""Multi-Arm Bandit Algorithms."""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_ARMS = 8
MU, SIGMA = 0.5, 0.1

np.random.seed(5678)
D = np.random.normal(MU, SIGMA, N_ARMS)
print(f"reward probabilities:\n{D}")
np.random.seed(None)


# plot distributions
x_axis = np.arange(0, 1, 0.001)
for i in range(N_ARMS):
    plt.plot(x_axis, norm.pdf(x_axis, D[i], SIGMA), label=f"arm_{i}")
plt.legend()
#plt.show()


def pull_arm(arm):
    """pull arm and get a reward."""
    noise = np.random.normal(0, 1)
    return D[arm] + noise


# Incremental Uniform
N_PULLS = 3000
bandit_pulls = [0 for _ in range(N_ARMS)]
estimates = np.zeros(N_ARMS)
total_rewards = 0
for _ in range(N_PULLS // N_ARMS):
    for arm in range(N_ARMS):
        reward = pull_arm(arm)
        bandit_pulls[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / bandit_pulls[arm]
        total_rewards += reward
best_bandit = estimates.argmax()
print(f"Incremental Uniform:\n\
bandit pulls={bandit_pulls}, \
best bandit={best_bandit}, \
estimate={estimates[best_bandit]}, \
rewards={int(total_rewards)}")

# epsilon-Greedy
N_PULLS = 3000
epsilon = 0.4
bandit_pulls = [0 for _ in range(N_ARMS)]
estimates = np.zeros(N_ARMS)
total_rewards = 0
for _ in range(N_PULLS):
    # choose best arm
    arm = estimates.argmax()
    # or choose other random arm with probability epsilon
    if epsilon > np.random.random():
        choices = list(range(N_ARMS))
        choices.remove(estimates.argmax())
        arm = np.random.choice(choices)
    # pull the arm
    reward = pull_arm(arm)
    bandit_pulls[arm] += 1
    estimates[arm] += (reward - estimates[arm]) / bandit_pulls[arm]
    total_rewards += reward
best_bandit = estimates.argmax()
print(f"epsilon greedy:\n\
bandit pulls={bandit_pulls}, \
best bandit={best_bandit}, \
estimate={estimates[best_bandit]}, \
rewards={int(total_rewards)}")

# UCB
N_PULLS = 3000
bandit_pulls = [0 for _ in range(N_ARMS)]
estimates = np.zeros(N_ARMS)
total_rewards = 0
for n in range(N_PULLS // N_ARMS):
    arm = 0
    max_upper_bound = 0
    for i in range(N_ARMS):
        upper_bound = 1e400
        if bandit_pulls[i] > 0:
            delta_i = math.sqrt(2 * math.log(n+1) / bandit_pulls[i])
            upper_bound = estimates[i] + delta_i

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            arm = i

        reward = pull_arm(arm)
        bandit_pulls[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / bandit_pulls[arm]
        total_rewards += reward
best_bandit = estimates.argmax()
print(f"UCB:\n\
bandit pulls={bandit_pulls}, \
best bandit={best_bandit}, \
estimate={estimates[best_bandit]}, \
rewards={int(total_rewards)}")
