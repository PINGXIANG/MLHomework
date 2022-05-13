"""
###############
##Tic-Tac-Toe##
###############

幻想用一个 program code n board size, k c.t.s. win 是愚蠢的。一个 program 的 extension 应当有一定的局限性才对。

Define
    state = [[1,2,3], [4,5,6], index]
where
    - The first and second lists are the location of the X's and O's respectively;
    - Index refer to the index of a large array holding the value of the state.
Note that without eliminating the cases when there exist both XXX and OOO, the total possible states is 6046.
The number is acceptable as the dict later on is easier to construct compared with the true number 5478.
    (len([]) = 0, len([])=0) -> comb(9,0)*comb(9,0)
    (len([]) = 1, len([])=0) -> comb(9,1),comb(8,0)
    (len([]) = 1, len([])=0) -> comb(9,1)*comb(8,1)
    ...
    (len([]) = 5, len([])=4) -> comb(9,5),comb(4,4)
Here, the order of index of the locations in the board does not matter. They are put into the box of X and O.
Hence, we can define the dict as
    state_values[(tuple(X_moves), tuple(O_moves))] -> value

Notice that each state can be reached even it is a terminal state. The only difference is the order to reach the state.
Hence, define the index of each state using the X-moves and O-moves as



Define the index of the location as follows:
    2 9 4
    7 5 3
    6 1 8
Then define the index of the location of the X's and the O's separately 2 arrays as, e.g.,
    1 2 3
    4 5 6
As long as the sum of the index = 15, then win.
Notice that we cannot have 2


    Value_table = -np.ones(6046)

TODO: As the loss value is 0 and win value is 1, the init value does not necessarily need to be None?
Otherwise it keeps on warning that None - None


English-Chines Dictionary:
    Bit - 落子

"""
from typing import List

import numpy as np
import itertools
from math import comb

magic_square = np.array([2, 9, 4, 7, 5, 3, 6, 1, 8])


def any_n_sum_to_k(moves: List[int], quiet=True):
    """
    Return True if there exist one XXX or OOO in the board. False if not.
    If set quiet = False, print ONLY one of the XXX or OOO index of location in the board.
    """
    result = False
    for sub_moves in itertools.combinations(moves, 3):
        if np.sum(sub_moves) == 15:
            if not quiet:
                print(sub_moves)

            result = True

    return result


def show_state(state, quiet=False):
    """
    Pretty_print.
    """
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


def init_value_table(quiet=True):
    """
    Init the value table.
    """
    value_table = {}

    board = list(range(1, 9 + 1))
    n = len(board)
    for k in range(1, 5 + 1):
        if not quiet:
            print(
                print("comb(9,%d)*comb(%d,%d)" % (k - 1, n - (k - 1), k - 1), comb(n, k - 1), comb(n - (k - 1), k - 1)))
        for X_moves in itertools.combinations(board, k - 1):
            for O_moves in itertools.combinations(list(set(board) - set(X_moves)), k - 1):
                value_table[(tuple(sorted(X_moves)), tuple(sorted(O_moves)))] = None

        if not quiet:
            print("comb(9,%d),comb(%d,%d)" % (k, n - k, k - 1), comb(n, k), comb(n - k, k - 1))
        for X_moves in itertools.combinations(board, k):
            for O_moves in itertools.combinations(list(set(board) - set(X_moves)), k - 1):
                value_table[(tuple(sorted(X_moves)), tuple(sorted(O_moves)))] = None

    return value_table


value_table = init_value_table()
init_state = [[], []]


def value(state):
    X_moves = tuple(sorted(state[0].copy()))
    O_moves = tuple(sorted(state[1].copy()))

    return value_table[(X_moves, O_moves)]


def set_value(state, value):
    X_moves = tuple(sorted(state[0].copy()))
    O_moves = tuple(sorted(state[1].copy()))

    value_table[(X_moves, O_moves)] = value


def next_state(player, state, move):
    """
    Return the new state after making the indicated move by the indicated player.
    Here, it is important to keep the terminal state with value 1 and 0 instead of 1.0 and 0.0, as the type is used to
    determine whether it is a terminal state or not.

    # This actually does not mater as both state will terminate before happens.
    By checking the X-moves first, this function make sure that the next state of (2,9,4), (5,7,3) is valued by 1.
    However, (2,9,1,4), (5,7,3) is valued at 1 as well, even though it has lost.
    :param player: 'X' or 'O'
    """
    X_moves = state[0].copy()
    O_moves = state[1].copy()
    if player == 'X':
        X_moves.append(move)
    elif player == "O":
        O_moves.append(move)

    next_state = [X_moves, O_moves]
    if value(next_state) == None:
        if any_n_sum_to_k(X_moves):
            set_value(next_state, 0)
        elif any_n_sum_to_k(O_moves):
            set_value(next_state, 1)
        elif len(X_moves) + len(O_moves) == 9:
            set_value(next_state, 0)
        else:
            set_value(next_state, 0.5)

    return next_state


def terminal_state_p(state):
    """
    If value of state is int, then it is terminal; otherwise, it is not a terminal
    """
    return isinstance(value(state), int)


alpha = 0.5
epsilon = 0.01


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


def random_move(state):
    """
    Return one of the unplaced locations, and selected at random
    """

    blanks = possible_moves(state)

    return np.random.choice(blanks)


def greedy_move(player, state):
    """
    Return the move which gives the highest valued position when played.
    """
    # Map the blanks and values using their index
    blanks = possible_moves(state)
    values = [value(next_state(player, state, move)) for move in blanks]

    # Obtain one action index of the max estimated action value.
    max_value = np.max(values)

    # Obtain all actions index of the max estimated action value.
    greedy_index_list = [index for index, value in enumerate(values) if value == max_value]

    greedy_index = np.random.choice(greedy_index_list)

    greedy_move = blanks[greedy_index]

    return greedy_move


# Now here is the main function
initial_state = [[], []]
value_table[((), ())] = 0.5


def game(quiet=False):
    """
    Plays 1 game against the random player. Also learns and prints.
    """
    state = initial_state
    show_state(state, quiet)

    while True:
        new_state = next_state('X', state, random_move(state))
        exploratory = np.random.binomial(n=1, p=epsilon)

        if terminal_state_p(new_state):
            show_state(new_state, quiet)

            update(state, new_state, quiet)

            return value(new_state)

        if exploratory:
            move = random_move(new_state)
        else:
            move = greedy_move('O', new_state)

        new_state = next_state('O', new_state, move)

        # TODO: Why I need a unless here?
        if not exploratory:
            update(state, new_state, quiet)
        show_state(new_state, quiet)

        if terminal_state_p(new_state):
            return value(new_state)

        state = new_state


def update(state, new_state, quiet=False):
    """
    This is the learning rule.
    """
    set_value(state, value(state) + alpha * (value(new_state) - value(state)))
    if not quiet:
        print(state, value(state))


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
    run()
    # runs(10, 40, 100)
