from typing import List
import itertools
import numpy as np


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