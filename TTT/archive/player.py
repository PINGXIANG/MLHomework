from typing import List

import numpy as np
import itertools
from math import comb
from assistant_func import any_n_sum_to_k
from board import Board

class Player:
    def __init__(self, player: str, board: Board):
        """
        value table: {state: float/int/None}. Int indicate a terminal state.
        state: [X_moves (list), O_moves (list)]
        moves: [Position (int)]
        :param player: 'X' or 'O'
        """
        self.player = player
        self.board = board
        self.value_table = self.init_value_table()

    def init_value_table(self, quiet=True):
        """
        Init the value table.
        """
        value_table = {}
        player = self.player

        def init_value(player, X_moves, O_moves):
            state = (tuple(sorted(X_moves)), tuple(sorted(O_moves)))
            if player == 'X':
                if any_n_sum_to_k(X_moves):
                    value_table[state] = 1
                elif any_n_sum_to_k(O_moves):
                    value_table[state] = -1
                elif len(X_moves) + len(O_moves) == 9:
                    value_table[state] = 0
                else:
                    value_table[state] = 0.5

            elif player == 'O':
                if any_n_sum_to_k(X_moves):
                    value_table[state] = -1
                elif any_n_sum_to_k(O_moves):
                    value_table[state] = 1
                elif len(X_moves) + len(O_moves) == 9:
                    value_table[state] = 0
                else:
                    value_table[state] = 0.5

        board = list(range(1, 9 + 1))
        n = len(board)
        # As the init only gets run for once at the beginning and only for 6_000 input, lets be silly here.
        for k in range(1, 5 + 1):
            if not quiet:
                print(
                    print("comb(9,%d)*comb(%d,%d)" % (k - 1, n - (k - 1), k - 1), comb(n, k - 1),
                          comb(n - (k - 1), k - 1)))
            for X_moves in itertools.combinations(board, k - 1):
                for O_moves in itertools.combinations(list(set(board) - set(X_moves)), k - 1):
                    init_value(player, X_moves, O_moves)

            if not quiet:
                print("comb(9,%d),comb(%d,%d)" % (k, n - k, k - 1), comb(n, k), comb(n - k, k - 1))
            for X_moves in itertools.combinations(board, k):
                for O_moves in itertools.combinations(list(set(board) - set(X_moves)), k - 1):
                    init_value(player, X_moves, O_moves)

        return value_table


    def value(self, state: List[List[int]]):
        """
        Return the value of the state based on value table.
        :param state: [X_moves, O_moves]
        """
        X_moves = tuple(sorted(state[0].copy()))
        O_moves = tuple(sorted(state[1].copy()))

        return self.value_table[(X_moves, O_moves)]

    def set_value(self, state: List[List[int]], value: float):
        """
        Modify the value of the state on the value table.
        :param state: [X_moves, O_moves]
        :param value: float
        """
        X_moves = tuple(sorted(state[0].copy()))
        O_moves = tuple(sorted(state[1].copy()))

        self.value_table[(X_moves, O_moves)] = value

    @staticmethod
    def next_state(player: str, state: List[List[int]], move: int):
        """
        Return the new state after making the indicated move by the indicated player.
        Here, it is important to keep the terminal state with value 1 and 0 instead of 1.0 and 0.0, as the type is used to
        determine whether it is a terminal state or not.

        # This actually does not mater as both state will terminate before happens.
        By checking the X-moves first, this function make sure that the next state of (2,9,4), (5,7,3) is valued by 1.
        However, (2,9,1,4), (5,7,3) is valued at 1 as well, even though it has lost.

        :param player:
        :type player:
        :param state: [X_moves, O_moves]
        :param value: float
        :param move: position index
        """

        X_moves = state[0].copy()
        O_moves = state[1].copy()

        if player == 'X':
            X_moves.append(move)
        elif player == "O":
            O_moves.append(move)

        next_state = [X_moves, O_moves]

        return next_state

    def terminal_state_p(self, state: List[List[int]]):
        """
        If value of state is int, then it is terminal; otherwise, it is not a terminal
        """
        return isinstance(self.value(state), int)


    def random_move(self, state):
        """
        Return one of the unplaced locations, and selected at random
        """

        blanks = self.board.possible_moves(state)

        return np.random.choice(blanks)

    def greedy_move(self, state):
        """
        Return the move of current player which gives the highest valued position when played given current state.
        """

        # Map the blanks and values using their index
        blanks = self.board.possible_moves(state)
        values = [self.value(self.next_state(state, move)) for move in blanks]

        # Obtain one action index of the max estimated action value.
        max_value = np.max(values)

        # Obtain all actions index of the max estimated action value.
        greedy_index_list = [index for index, value in enumerate(values) if value == max_value]

        greedy_index = np.random.choice(greedy_index_list)

        greedy_move = blanks[greedy_index]

        return greedy_move

    def update(self, state, new_state, alpha: float, quiet=False):
        """
        This is the learning rule.
        """
        state = self.board.state
        self.set_value(state, self.value(state) + alpha * (self.value(new_state) - self.value(state)))
        if not quiet:
            print(state, self.value(state))

    def next_move(self, state, epsilon: float):
        exploratory = np.random.binomial(n=1, p=epsilon)

        if exploratory:
            move = self.random_move()
        else:
            move = self.greedy_move()

        new_state = self..next_state(new_state, move)

        # TODO: Why I need a unless here?
        if not exploratory:
            update(state, new_state, quiet)
        show_state(new_state, quiet)