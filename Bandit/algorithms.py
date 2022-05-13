from typing import List

import numpy as np
import random
from bandit_model import Bandit, ModifiedBandit


class Algorithm:
    def __init__(self, model: Bandit, init_action_value: float = 0.0):
        self.model = model
        self.history = [
            [],  # Action At for each t
            [],  # Reward Rt for each t
        ]

        # Interpret the count list as a map from index i to action a_i. Init all to be 0.
        self.count_list = [0] * len(self.model.action_set)

        # Interpret the estimated action value list as a map from index i to action i. Init all to be 0.
        self.estimated_action_value_list = [init_action_value] * len(self.model.action_set)

    def obtain_count(self, t: int, action: int):
        """
        return the count of action a_i at time step t. The math notation is
            N_t(a) = sum_{i=1}^{t-1} 1_{A_i = a}
        Here we will note it as Nt_a.
        :return: Nt_a
        :rtype: int
        """
        try:
            Nt_a = np.sum(self.history[0][:t] == action)
        except IndexError as e:
            print(e)
            Nt_a = 0

        return Nt_a

    def update_count(self, action: int):
        """
        Update the count of action a_i at time step t. The math notation is
            N_t(a) = sum_{i=1}^{t-1} 1_{A_i = a}
        Here we will note it as Nt_a.
        :return: None
        """
        # Only modify the count Nt(a) for the action At = a performed.
        action_index = self.model.action_set.index(action)
        self.count_list[action_index] += 1

    def estimate_action_value(self, t: int, action):
        """
        Return the estimated action value of a_i at time t. The math notation is
            Q_{t+1}(a) = Qt(a) + [Rt-Qt(a)] / Nt(a)
        Here we note it as Qt_a, where Nt_a is the count of the action a_i at time t.
        :return: Qt_a
        :rtype: float
        """
        # Only estimate the action At = a performed.
        action_index = self.model.action_set.index(action)
        Nt_a = self.count_list[action_index]
        reward = self.history[1][-1]
        Qt_a = self.estimated_action_value_list[action_index]
        Qt_a += (reward - Qt_a) / Nt_a
        return Qt_a

    def update_estimated_action_value(self, t: int, action):
        """
        Update the estimated action value of a_i at time t. The math notation is
            Q_{t+1}(a) = Qt(a) + [Rt-Qt(a)] / Nt(a)
        Here we note it as Qt_a, where Nt_a is the count of the action a_i at time t.
        :return: None
        """
        # Only modify the estimated action value Qt(a) for the action At = a performed.
        Qt_a = self.estimate_action_value(t, action)
        action_index = self.model.action_set.index(action)
        self.estimated_action_value_list[action_index] = Qt_a

    def get_greedy_index_list(self) -> List[int]:
        """
        Return the greedy action a* at time t.
        :return: greedy action list[], len>=1
        :rtype: List[int]
        """
        # Obtain one action index of the max estimated action value at time t.
        max_value = max(self.estimated_action_value_list)

        # Obtain all actions index of the max estimated action value at time t.
        greedy_index_list = [index for index, value in enumerate(self.estimated_action_value_list) if
                             value == max_value]

        return greedy_index_list

    def take_action(self, t: int, action: int):
        """
        Update the action and reward to the end of the history list.
        """
        # Take action At
        self.history[0].append(action)
        self.update_count(action)

        # Get reward Rt
        reward = self.model.simulate_reward(t, action)
        self.history[1].append(reward)
        self.update_estimated_action_value(t, action)

    def select_action(self, t: int):
        pass


class SampleAvg(Algorithm):
    def estimate_action_value(self, t: int, action: int):
        """
        Return the estimated action value of a_i at time t. The math notation is
            Q_{t+1}(a) = Qt(a) + [Rt-Qt(a)] / Nt(a)
        Here we note it as Qt_a, where Nt_a is the count of the action a_i at time t.
        :return: Qt_a
        :rtype: float
        """
        action_index = self.model.action_set.index(action)
        Nt_a = self.count_list[action_index]
        reward = self.history[1][-1]
        Qt_a = self.estimated_action_value_list[action_index]
        Qt_a += (reward - Qt_a) / Nt_a
        return Qt_a


class EpsilonGreedyAlgorithm(Algorithm):
    def learn(self):
        pass

    def select_action(self, t: int, epsilon: float = 0.0):

        # Split the action set into greedy and non-greedy
        action_set = self.model.action_set.copy()
        greedy_action_list = []
        non_greedy_action_list = []
        greedy_index_list = self.get_greedy_index_list()
        for i in range(len(action_set)):
            if i in greedy_index_list:
                greedy_action_list.append(action_set[i])
            else:
                non_greedy_action_list.append(action_set[i])

        # Randomize the action selection 
        non_greedy = np.random.binomial(n=1, p=epsilon)
        if non_greedy:
            action = random.choice(greedy_action_list)
        else:
            action = random.choice(non_greedy_action_list)

        return action
