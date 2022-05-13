from board import Board
from player import Player
import numpy as np

alpha = 0.5
epsilon = 0.01

def game(quiet=False):
    """
    Plays 1 game against the random player. Also learns and prints.
    """
    board = Board()
    p1 = Player('X', board)
    p2 = Player('O', board)
    board.show_state(quiet)

    while True:
        new_state = p1.next_state(board.state, p1.random_move())

        # If it is the terminal state, then show the new state. Clearly this is the greedy action. No need to play.
        if terminal_state_p(new_state):
            board.show_state(new_state, quiet)

            p1.update(new_state, alpha, quiet)

            return p1.value(new_state)

        # Next move
        move = p2.next_move(epsilon)

        new_state = p2.next_state(new_state, move)

        # TODO: Why I need a unless here?
        if not exploratory:
            update(state, new_state, quiet)
        show_state(new_state, quiet)

        if terminal_state_p(new_state):
            return value(new_state)

        state = new_state





def run():
    for _ in range(40):
        v = 0
        for _ in range(100):
            v += game()

        print(v / 100.0)


def runs(num_runs, num_bins, bin_size):
    array = np.zeros(num_bins)
    for _ in range(num_runs):
        for i in range(num_bins):
            for _ in range(bin_size):
                array[i] += game()

        for i in range(num_bins):
            print(array[i] / (bin_size * num_runs))


if __name__ == "__main__":
    runs(10, 40, 100)
