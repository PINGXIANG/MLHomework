import numpy as np

k = 10


class Agent:
    def __init__(self, Q_0=0, c=0, epsilon=0.):
        self.Q = np.zeros(k) + Q_0  # The estimated value for each action
        self.N_a = np.zeros(k)
        self.c = c
        self.epsilon = epsilon

    def learn(self, a: int, r: float, alpha=0.1):
        self.N_a[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N_a[a]

    def greedy(self, t: int):
        if self.c == 0:
            return np.argmax(self.Q)
        else:
            return np.argmax(self.Q + self.c * np.sqrt(np.log(t) / self.N_a))

    def epsilon_greedy(self, t: int):
        if np.random.binomial(n=1, p=self.epsilon):
            action = np.random.choice(k)  # randomly select one of the 10 beds
        else:
            action = self.greedy(t)

        return action


class SampleMeanAgent(Agent):
    def __init__(self, Q_0=0, c=0, epsilon=0.):
        super(SampleMeanAgent, self).__init__(Q_0, c, epsilon)
        self.rewards = [[] for _ in range(k)]

    def learn(self, a: int, r: float, alpha=0.1):
        self.N_a[a] += 1
        self.rewards[a].append(r)
        self.Q[a] = np.array(self.rewards[a]).mean()


class StepSizeAgent(Agent):
    def learn(self, a: int, r: float, alpha=0.1):
        self.Q[a] += alpha * (r - self.Q[a])
