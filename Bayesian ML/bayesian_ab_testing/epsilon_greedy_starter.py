# epsilon-greedy starter algorithm

import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1 # Epsilon: Exploration Factor
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75] # Actual win rates of bandits (unknown in reality)

class Bandit:
    def __init__(self, p):
        self.p = p # p represents the true win rate for this bandit
        self.p_estimate = 0 
        self.N = 0 # number of samples collected

    def pull(self): # Pulling lever of 'slot machine'
        return np.random.random() < self.p # Return True(1) if win / False(0) if lose

    def update(self, x):
        self.N = self.N + 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N # Iteratively updating sample mean (p_estimate)

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax(([b.p for b in bandits])) # np.argmax() returns the 'indices' if the maximum values along an axis
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])
        
        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)


    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_times_explored)
    print("num_times_exploited:", num_times_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
    experiment()