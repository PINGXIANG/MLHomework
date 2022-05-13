from assistant_func import *
import assistant_func
player_dict = {
    'X': 0,
    'O': 1
}

class Agent:
    def __init__(self, player: str, value_table: dict, player_type: str):
        self.player = player
        self.player_type = player_type

        if self.player == 'X':
            self.opponent = 'O'
        elif self.player == 'O':
            self.opponent = 'X'
        self.value_table = value_table

    # One can only update his own value table
    def value(self, state):
        X_moves = tuple(sorted(state[0].copy()))
        O_moves = tuple(sorted(state[1].copy()))

        return self.value_table[(X_moves, O_moves)]

    def set_value(self, state, value):
        X_moves = tuple(sorted(state[0].copy()))
        O_moves = tuple(sorted(state[1].copy()))

        self.value_table[(X_moves, O_moves)] = value

    def update(self, state, new_state, quiet=False):
        """
        This is the learning rule.
        """
        pass


class Player(Agent):
    def __init__(self, player: str, value_table: dict, player_type: str):
        super().__init__(player, value_table, player_type)

    # Describe how the policy of the agent looks like
    def move(self, state, epsilon: float):
        blanks = possible_moves(state)
        exploratory = np.random.binomial(n=1, p=epsilon)

        if exploratory:
            move = np.random.choice(blanks)
        else:
            move = self.greedy_move(state)

        return move, exploratory

    def greedy_move(self, state):
        """
        Return the move which gives the highest valued position when played.
        """
        # Map the blanks and values using their index
        blanks = possible_moves(state)
        values = [self.value(next_state(self.player, state, move)) for move in blanks]

        # Obtain one action index of the max estimated action value.
        max_value = np.max(values)

        # Obtain all actions index of the max estimated action value.
        greedy_index_list = [index for index, value in enumerate(values) if value == max_value]

        greedy_index = np.random.choice(greedy_index_list)

        greedy_move = blanks[greedy_index]

        return greedy_move

    def update(self, state, new_state, quiet=False):
        """
        This is the learning rule.
        """
        self.set_value(state, self.value(state) + alpha * (self.value(new_state) - self.value(state)))
        if not quiet:
            print(state, self.value(state))

class Opponent(Agent):
    def __init__(self, player: str, value_table: dict, player_type: str):
        super().__init__(player, value_table, player_type)

    # Describe how the policy of the agent looks like
    def move(self, state, epsilon: float):
        exploratory = 0
        blanks = possible_moves(state)

        if self.player_type == 'type 1':
            next_move = np.random.choice(blanks)

            return next_move, exploratory

        elif self.player_type == 'type 2':
            # values1 = [self.value(assistant_func.next_state(self.player, state, move)) for move in blanks]
            # if 1 in values1:
            #     move_index = values1.index(1)
            #     next_move = blanks[move_index]
            # else:
            #     next_move = np.random.choice(blanks)
            for move in blanks:
                next_state = assistant_func.next_state(self.player, state, move)
                player_moves = next_state[player_dict[self.player]].copy()
                opponent_moves = next_state[player_dict[self.opponent]].copy()
                if any_n_sum_to_k(player_moves):
                    next_move = move

                    return next_move, exploratory

            next_move = np.random.choice(blanks)
            return next_move, exploratory

        # TODO: Not sure the best way to implement it
        elif self.player_type == 'type 3':
            for move in blanks:
                next_state = assistant_func.next_state(self.player, state, move)
                player_moves = next_state[player_dict[self.player]].copy()
                opponent_moves = next_state[player_dict[self.opponent]].copy()
                if any_n_sum_to_k(player_moves):
                    next_move = move
                    return next_move, exploratory

            for move in blanks:
                next_state = assistant_func.next_state(self.opponent, state, move)
                player_moves = next_state[player_dict[self.player]].copy()
                opponent_moves = next_state[player_dict[self.opponent]].copy()
                if any_n_sum_to_k(opponent_moves):
                    next_move = move
                    return next_move, exploratory

            next_move = np.random.choice(blanks)

            return next_move, exploratory


