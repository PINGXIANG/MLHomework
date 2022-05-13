import numpy as np


# Bandit for finite time steps and finite action sets.
class Bandit:
    def __init__(self, action_set: list, time_steps: int):
        self.n_arms = len(action_set)
        self.time_steps = time_steps
        # The order of action set is important as I will map it to a range of coordinate
        self.action_set = action_set
        # Set up the action value vector at the beginning
        self.action_value_vec = self.simulate_action_value_vec()

    def get_action_value(self, action: int):
        """
        return the true action value of action a at time t.
        """

        return self.action_value_vec[self.action_set.index(action), 0]

    def simulate_action_value_vec(self, seed=None):
        """
        simulate the true action values of each action a. Note that in the k-armed testbed setting, the action value is
        fixed over time steps t.
        :rtype: np.array(n_actions, 1)
        """
        np.random.seed(seed)

        return np.random.normal(size=(self.n_arms, 1))

    def simulate_reward(self, t: int, action: int, seed=None):
        """
        return the reward for EACH action at time t.
        :rtype: np.array(n_actions, 1)
        """
        np.random.seed(seed)

        reward = np.random.normal(loc=self.get_action_value(action), scale=1.0)

        return reward


class ModifiedBandit(Bandit):

    def simulate_action_value_vec(self):
        """
        return the true action value of action a at time t.
        """
        action_value_matrix = np.ones(shape=(self.n_arms, self.time_steps))
        action_value_matrix += np.cumsum(np.random.normal(size=(self.n_arms, self.time_steps)), axis=1)

        return action_value_matrix

