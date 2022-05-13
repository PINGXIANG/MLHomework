from typing import List

import numpy as np

# private
magic_square = np.array([2, 9, 4, 7, 5, 3, 6, 1, 8])
init_state = [[], []]


class Board:
    def __init__(self):
        self.state = init_state

    def show_state(self, quiet=False):
        """
        Pretty_print.
        """
        state = self.state.copy()
        if not quiet:
            X_moves = state[0].copy()
            O_moves = state[1].copy()
            dig_board = list(magic_square)
            board = list(magic_square.astype(str))

            for X in X_moves:
                X_i = dig_board.index(X)
                board[X_i] = 'X'
            for O in O_moves:
                O_i = dig_board.index(O)
                board[O_i] = 'O'

            for i in range(3):
                print("  ".join(board[3 * i: 3 * (i + 1)]))

    @staticmethod
    def possible_moves(state):
        """
        Return the possible moves. Here it is denoted as blanks.
        """

        blanks = []
        for i in range(1, 9 + 1):
            X_moves = state[0].copy()
            O_moves = state[1].copy()
            if i not in X_moves and i not in O_moves:
                blanks.append(i)

        return blanks

