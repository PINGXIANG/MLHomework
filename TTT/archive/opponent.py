from TTT.archive.board import Board
from board import Board
import numpy as np
from assistant_func import any_n_sum_to_k
from player import Player


class Opponent(Player):

    def __init__(self, player: str, opponent_type: str, board: Board):
        super().__init__(player, board)
        self.opponent_type = opponent_type

    def next_move(self, state):
        blanks = self.board.possible_moves()
        opponent_type = self.opponent_type
        player = self.player

        if opponent_type == 'type 1':
            next_move = np.random.choice(blanks)
            return next_move
        elif opponent_type == 'type 2':
            values = [O_win(self.next_state(player, state, move)) for move in blanks]
            move_index = values.index(1)
            next_move = blanks[move_index]
            return next_move
        elif opponent_type == 'type 3':
            # for move in blanks:
            #     next_state = self.next_state(move)
            #     X_blanks = self.board.possible_moves()
            #     for X_move in X_blanks:
            #         X_state = self.
            #         value = self.value()
            pass

    @staticmethod
    def X_win(state):
        X_moves = state[0].copy()
        O_moves = state[1].copy()

        if any_n_sum_to_k(X_moves):
            return True
        else:
            return False

    @staticmethod
    def O_win(state):
        O_moves = state[1].copy()

        if any_n_sum_to_k(O_moves):
            return True
        else:
            return False
