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
from agent import Agent
import matplotlib.pyplot as plt

max_num_tasks = 2000

# Set up
k = 10

# Q_star follows a random walk
Q_star = np.concatenate((np.zeros(shape=(k,1)), np.random.normal(size=(k, 10))), axis=1).cumsum(axis=1)


def reward(a, task_num):
    """
    There is a random walk for each reward.
    """
    return Q_star[a][task_num] + np.random.normal()

def max_Q_star():
    return Q_star.argmax(axis=0)

def run(agent=Agent(), num_runs=1000, num_steps=100, epsilon=0.):
    avg_reward = np.zeros(num_steps)
    prob_a_star = np.zeros(num_steps)
    optimal_action = max_Q_star()
    for run_num in range(num_runs):  # At game run_num
        a_star = optimal_action[run_num]

        for time_step in range(num_steps):
            a = agent.epsilon_greedy(t=time_step)
            r = reward(a, run_num)
            agent.learn(a, r)
            avg_reward[time_step] += r
            if a_star == a:
                prob_a_star[time_step] += 1

    avg_reward = avg_reward / num_runs
    prob_a_star = prob_a_star / num_runs

    return avg_reward, prob_a_star


if __name__ == "__main__":
    agent_0 = Agent(Q_0=5, c=0, epsilon=0)
    agent_01 = Agent(Q_0=0, c=2, epsilon=0.1)
    agent_001 = Agent(Q_0=0, c=2, epsilon=0.01)

    avg_reward_0, prob_a_star_0 = run(agent=agent_0, num_runs=100, num_steps=1000)
    avg_reward_01, prob_a_star_01 = run(agent=agent_01, num_runs=100, num_steps=1000)
    avg_reward_001, prob_a_star_001 = run(agent=agent_01, num_runs=100, num_steps=1000)

    plt.plot(avg_reward_0, label="\epsilon$ = 0")
    plt.plot(avg_reward_01, label="\epsilon$ = 0.1")
    plt.plot(avg_reward_001, label="\epsilon$ = 0.01")
    # plt.plot(RMSE_Bellman_2, label = "Bellman 2")
    plt.xlabel('Steps')
    plt.ylabel('avg reward')
    plt.legend()
    plt.savefig('avg reward')

    plt.plot(prob_a_star_0, label="$\epsilon$ = 0")
    plt.plot(prob_a_star_01, label="$\epsilon$ = 0.1")
    plt.plot(prob_a_star_001, label="$\epsilon$ = 0.01")
    # plt.plot(RMSE_Bellman_2, label = "Bellman 2")
    plt.xlabel('Steps')
    plt.ylabel('optimal action %')
    plt.legend()
    plt.savefig('optimal action %')