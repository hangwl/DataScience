# epsilon-greedy (and comparing different epsilon value outcomes)
import matplotlib.pyplot as plt
import numpy as np

# OUTLINE:
# 1. define BanditArm class
# 2. define function that runs experiment over multiple trials
# 3. print and plot results

# the BanditArm class has two functions
# 1. pull - which returns the reward m which follows a gaussian distribution
# 2. update - update the trial counter and m_estimate (sample mean of reward) for the respective bandit explored/exploited

class BanditArm: 
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m # in this example, we are drawing a sample from a gaussian distribution with mean m and variance 1 rather than a bernoulli

    def update(self, x):
        self.N += 1
        self.m_estimate = (1 - 1.0/self.N)*self.m_estimate + 1.0/self.N*x # the way to calculate sample mean remains unchanged

# in the epsilon-greedy approach, our experiment will be iterated over 3 bandits with their individual respective means m
# epsilon is defined as the probability that a random bandit will be explored, else we exploit the bandit with the highest estimated m_estimate
# over each exploration/exploitation, we will update the respective m_estimate

def run_experiment(m1, m2, m3, eps, N):
    bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]

    # count number of suboptimal choices
    means = np.array([m1,m2,m3])
    true_best = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)

    for i in range(N):
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.m_estimate for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        if j != true_best:
            count_suboptimal += 1

        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1) # np.cumsum() returns an array of consecutive cumulative reward averages i.e. np.cumsum([1,2,3] returns array([1,3,6])) 

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log') # these algorithms converge quickly, and using a log scale is useful in this case
    plt.show()

    for b in bandits:
        print(b.m_estimate)

    print("percent suboptimal for epsilon = %s:" % eps, float(count_suboptimal) / N)

    return cumulative_average

# in this exercise, we run our experiments over different epsilon values
# when we compare all 3 plots together, we notice:
# higher epsilon means that the algorithm converges more quickly to the reward of the optimal bandit;
# when we compare the long run cumulative reward averages, we notice that for higher epsilon values, the cumulative reward is worse and worse
# this implies that there is a trade-off between fast convergence and higher eventual reward

if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()

