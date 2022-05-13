import numpy as np
import re


s = 'XXOOX XOO'

# state2str = {
#     0: ' ',
#     1: 'X',
#     -1: 'O'
# }


player2str = {
    0: 'X',
    1: 'O'
}


class Board:
    def __init__(self, board_size: int = 3):
        # terminate if there is # of k continuous same X or O. k = 3 is Tic-Tac-Tak while k = 5 is GoBang. 
        self.k = 3
        self.board_size = board_size
        self.current_state = ' ' * board_size
        self.history = [2]  # Position.point (int, int) Position.index int
        self.current_player = 0

    def is_terminate(self):
        """
        Return:
            None - Not terminate
            1 - X win
            0 - Draw/Stalemate (和棋)
           -1 - X lose, i.e., O win

        So one can use:
            while self.is_terminate not None
        to process the game and simultaneously knows the result.
        """

        k = self.k
        n = self.board_size
        current_state = self.current_state

        position = self.history[-1]  # index (int, int)
        x, y = self.index2point(position)

        left_end, right_end = np.max([x - (k - 1), 0]), np.min([x + (k - 1), n - 1])
        horizontal = current_state[self.point2index(left_end, y): self.point2index(right_end, y)]
        if 'X' * k in horizontal:
            return 1
        elif 'O' * k in horizontal:
            return -1

        top_end, bottom_end = np.max([y - (k - 1), 0]), np.min([y + (k - 1), n - 1])
        vertical = ''
        for j in range(top_end, bottom_end + 1):
            vertical += current_state[self.point2index(x, j)]
        if 'X' * k in vertical:
            return 1
        elif 'O' * k in vertical:
            return -1

        diagonal_topLeft2bottomRight = ''
        topLeft_delta, bottomRight_delta = np.min([x - left_end, y - top_end]), np.min([right_end - x, bottom_end - y])
        for l in range(topLeft_delta + bottomRight_delta + 1):
            diagonal_topLeft2bottomRight += current_state[
                self.point2index(x - bottomRight_delta + l, y - bottomRight_delta + l)]
        if 'X' * k in diagonal_topLeft2bottomRight:
            return 1
        elif 'O' * k in diagonal_topLeft2bottomRight:
            return -1

        diagonal_bottomLeft2TopRight = ''
        bottomLeft_delta, TopRight_delta = np.min([x - left_end, y - bottom_end]), np.min([right_end - x, top_end - y])
        for l in range(bottomLeft_delta + TopRight_delta + 1):
            diagonal_bottomLeft2TopRight += current_state[
                self.point2index(x - bottomLeft_delta + l, y - bottomLeft_delta + l)]
        if 'X' * k in diagonal_bottomLeft2TopRight:
            return 1
        elif 'O' * k in diagonal_bottomLeft2TopRight:
            return -1

        # None of X or O win, and there is no space for next state
        if ' ' not in current_state:
            return 0
        else:
            return None

    def put(self, position: int):
        current_player = self.current_player
        blank_positions = self.get_blanks()
        if position in blank_positions:  # Note that position is not in empty list anyway
            self.update_state(current_player, position)
            self.history.append(position)
            self.update_current_player()
            # Return 0
        else:
            raise Exception("Invalid position input.")
            # Return 1

    def get_blanks(self):
        """
        Return a list consists of all position of blank at current state.
        """
        return [m.start() for m in re.finditer(' ', self.current_state)]

    def update_current_player(self):
        return len(self.history) % 2

    def update_state(self, current_player, position):
        current_state = self.current_state
        state = current_state[:position] + player2str[current_player] + s[position + 1:]

        return state

    @staticmethod
    def pretty_print(state: str):
        size = int(np.sqrt(len(state)))
        for i in range(size):
            print("  ".join(state[size * i:size * (i + 1)]))

    def index2point(self, index: int):
        return (index - index % self.board_size), index % self.board_size

    def point2index(self, i: int, j: int):
        return i * self.board_size + j
