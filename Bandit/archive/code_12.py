"""
n - number of bandit
action = 0,1,2,3,4,5,6,7,8
epsilon = 0.1
Q* - The real value of action (10, 2000)
Q - The estimated value for each action (10,)
N_a - The # of action i (10,)
randomness
max_num_tasks = 2000
"""
import numpy as np

max_num_tasks = 2000

# Set up
n = 10

Q = np.zeros(n)  # The estimated value for each action
N_a = np.zeros(n)
Q_star = np.random.normal(size=(n, max_num_tasks))
randomness = np.random.normal(max_num_tasks)


def learn(a, r):
    N_a[a] += 1
    Q[a] += (r - Q[a]) / N_a[a]


def reward(a, task_num):
    """
    There is a random walk for each reward.
    """
    return Q_star[a][task_num] + np.random.normal()


def greedy():
    return np.argmax(Q)


def epsilon_greedy(epsilon):
    if np.random.binomial(n=1, p=epsilon):
        action = np.random.choice(n)  # randomly select one of the 10 beds
    else:
        action = greedy()

    return action


def max_Q_star():
    return Q_star.argmax(axis=0)


def run(num_runs=1000, num_steps=100, epsilon=0):
    avg_reward = np.zeros(num_steps)
    prob_a_star = np.zeros(num_steps)
    for run_num in range(num_runs):
        a_star = 0
        for a in range(1, n):
            if Q_star[a][run_num] > Q_star[a_star][run_num]:
                a_star = a

        random_state = randomness[run_num]

        collect = []
        for time_step in range(num_steps):
            a = epsilon_greedy(epsilon)
            r = reward(a, run_num)
            learn(a, r)
            avg_reward[time_step] += r
            if a_star == a:
                prob_a_star[time_step] += 1

        for i in range(num_steps):
            avg_reward[i] = avg_reward[i] / num_runs
            prob_a_star[i] = prob_a_star[i] / num_runs


if __name__ == "__main__":
    run()
