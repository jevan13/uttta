from utils import State, Action, load_data
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from torch import tensor
import time

class StudentAgent:
    def __init__(self):
        self.states = None
        self.global_boards = None
        self.local_boards = None
        self.values = None

    def test(self):
        states = []
        values = []

        data = load_data()
        for state, value in data:
            states.append(state)
            values.append(value)
            
        self.states = np.array(states)
        self.global_boards = np.array([state.local_board_status for state in states])
        self.local_boards = np.array([state.board.reshape(9, 3, 3) for state in states])
        self.values = np.array(values)

    def get_features(self):
        features = []
        
        # global centres
        global_centres = np.array([self.count_centre(board, 1) for board in self.global_boards])
        
        # global corners
        global_corners = np.array([self.count_corners(board, 1) for board in self.global_boards])
        
        # global sides
        global_sides = np.array([self.count_sides(board, 1) for board in self.global_boards])
        
        # local centres
        local_centres = np.array([sum(self.count_centre(each_local_board, 1)
                                      for each_local_board in sample)
                                      for sample in self.local_boards])
        
        # local corners
        local_corners = np.array([sum(self.count_corners(each_local_board, 1)
                                      for each_local_board in sample)
                                      for sample in self.local_boards])
        
        # local sides
        local_sides = np.array([sum(self.count_sides(each_local_board, 1)
                                      for each_local_board in sample)
                                      for sample in self.local_boards])

        # global three in a row
        global_three_in_a_row = np.array([self.global_three_in_a_row(state, 1) for state in self.states])

        # gloal two in a row
        global_two_in_a_row = np.array([self.x_in_a_row(board, 1, 2) for board in self.global_boards])

        # global one in a row
        global_one_in_a_row = np.array([self.x_in_a_row(board, 1, 1) for board in self.global_boards])

        # local three in a row
        local_three_in_a_row = np.array([self.local_three_in_a_row(state.local_board_status, 1) for state in self.states])

        # local two in a row
        local_two_in_a_row = np.array([sum(self.x_in_a_row(each_local_board, 1, 2)
                                      for each_local_board in sample)
                                      for sample in self.local_boards])
        
        # local one in a row
        local_one_in_a_row = np.array([sum(self.x_in_a_row(each_local_board, 1, 1)
                                      for each_local_board in sample)
                                      for sample in self.local_boards])
        
        # free board movement
        free_movement = np.array([self.has_free_movement(state) for state in self.states])
                        
        features = np.column_stack([
            global_centres,
            global_corners,
            global_sides,
            local_centres,
            local_corners,
            local_sides,
            global_three_in_a_row,
            global_two_in_a_row,
            global_one_in_a_row,
            local_three_in_a_row,
            local_two_in_a_row,
            local_one_in_a_row,
            free_movement
            ])
        return features

    def train_model(self):
        features = self.get_features()   
        labels = self.values
        net = nn.Linear(13, 1)
        self.train(net, features, labels)
        torch.set_printoptions(precision=10)
        print(net.state_dict())

    def train(self, net: nn.Module, X: np.ndarray, y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y[:, None], dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            output = net(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    def count_centre(self, board: np.array, player: 1 | 2):
        if board[1][1] == player:
            return 1
        elif board[1][1] == 3 - player:
            return -1
        else:
            return 0

    def count_corners(self, board: np.array, player: 1 | 2):
        count = 0
        for (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if board[i][j] == player:
                count += 1
            elif board[i][j] == 3 - player:
                count -= 1
        return count

    def count_sides(self, board: np.array, player: 1 | 2):
        count = 0
        for (i, j) in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if board[i][j] == player:
                count += 1
            elif board[i][j] == 3 - player:
                count -= 1
        return count
            
    def global_three_in_a_row(self, state: State, player: 1 | 2):
        if state.is_terminal():
            terminal_utility = state.terminal_utility()
            if (terminal_utility == 1.0 and player == 1) or (terminal_utility == 0.0 and player == 2):
                return 1
            elif (terminal_utility == 1.0 and player == 2) or (terminal_utility == 0.0 and player == 1):
                return -1
            else:
                return 0
        else:
            return 0

    def local_three_in_a_row(self, board: np.array, player: 1 | 2):
        return np.count_nonzero(board == player) - np.count_nonzero(board == 3 - player)

    def x_in_a_row(self, board: np.array, player: 1 | 2, x: int):
        opponent = 3 - player
        count = 0

        for row in board:
            count_row_player = np.count_nonzero(row == player)
            count_row_opponent = np.count_nonzero(row == opponent)
            count_row_zero = np.count_nonzero(row == 0)
            if count_row_player == x and count_row_zero == 3 - x:
                count += 1
            if count_row_opponent == x and count_row_zero == 3 - x:
                count -= 1

        for col in np.transpose(board):
            count_col_player = np.count_nonzero(col == player)
            count_col_opponent = np.count_nonzero(col == opponent)
            count_col_zero = np.count_nonzero(col == 0)
            if count_col_player == x and count_col_zero == 3 - x:
                count += 1
            if count_col_opponent == x and count_col_zero == 3 - x:
                count -= 1

        board_diag = np.diag(board)
        count_diag_player = np.count_nonzero(board_diag == player)
        count_diag_opponent = np.count_nonzero(board_diag == opponent)
        count_diag_zero = np.count_nonzero(board_diag == 0)
        if count_diag_player == x and count_diag_zero == 3 - x:
            count += 1
        if count_diag_opponent == x and count_diag_zero == 3 - x:
            count -= 1

        other_board_diag = np.diag(np.fliplr(board))
        count_other_diag_player = np.count_nonzero(other_board_diag == player)
        count_other_diag_opponent = np.count_nonzero(other_board_diag == opponent)
        count_other_diag_zero = np.count_nonzero(other_board_diag == 0)
        if count_other_diag_player == x and count_other_diag_zero == 3 - x:
            count += 1
        if count_other_diag_opponent == x and count_other_diag_zero == 3 - x:
            count -= 1

        return count

    def has_free_movement(self, state: State) -> bool:
        prev_action = state.prev_local_action
        if prev_action is None:
            return True
        row, col = prev_action
        return (state.local_board_status[row][col] != 0).astype(int)

    def choose_action(self, state: State) -> Action:
        max_depth = 10
        best_action = None
        start_time = time.time()
        time_limit = start_time + 2.9
        my_player_number = state.fill_num
        for depth in range(4, max_depth + 1):
            print("depth = ", depth)
            best = self.max_value(state, -np.inf, np.inf, depth, time_limit, my_player_number)
            if best[2] is not None:
                best_action = best[2]
            if time.time() > time_limit:
                break
        return best_action

    def max_value(self, state: State, alpha: int, beta: int, depth: int, time_limit: float, player: 1 | 2):
        if state.is_terminal() or depth == 0 or time.time() > time_limit:
            return (self.heuristic_func(state, player), state, None)
        else:
            best_v = -np.inf
            best_state = state
            best_action = None
            for next_action in state.get_all_valid_actions():
                if time.time() > time_limit:
                    break
                next_state = state.change_state(next_action, None, False, True)
                next_v = self.min_value(next_state, alpha, beta, depth - 1, time_limit, player)[0]
                if next_v > best_v:
                    best_v = next_v
                    best_state = next_state
                    best_action = next_action
                    alpha = max(alpha, best_v)
                    if best_v >= beta:
                        return (best_v, best_state, best_action)
            return (best_v, best_state, best_action)

    def min_value(self, state: State, alpha: int, beta: int, depth: int, time_limit: float, player: 1 | 2):
        if state.is_terminal() or depth == 0 or time.time() > time_limit:
            return (self.heuristic_func(state, player), state, None)
        else:
            best_v = np.inf
            best_state = state
            best_action = None
            for next_action in state.get_all_valid_actions():
                if time.time() > time_limit:
                    break
                next_state = state.change_state(next_action, None, False, True)
                next_v = self.max_value(next_state, alpha, beta, depth - 1, time_limit, player)[0]
                if next_v < best_v:
                    best_v = next_v
                    best_state = next_state
                    best_action = next_action
                    beta = min(beta, best_v)
                    if best_v <= alpha:
                        return (best_v, best_state, best_action)
            return (best_v, best_state, best_action)

    def heuristic_func(self, state: State, player: 1 | 2):
        opponent = 3 - player
        count = 0

        if state.is_terminal():
            term_util = state.terminal_utility()
            if term_util == 1.0:
                if player == 1:
                    return 100000
                else:
                    return -100000
            elif term_util == 0.0:
                if player == 1:
                    return -100000
                else:
                    return 100000
            else:
                return 0
        else:
            weights = np.array([-2.02301552e-01, -1.44824009e-01, -8.83835924e-02, 5.13033473e+09,
                        5.13033473e+09, 5.13033473e+09, 1.18988249e+00, 4.08901778e-01,
                        1.39618277e-01, -5.13033473e+09,  7.16277592e-02,  2.77853470e-02,
                        -5.25132319e-03])
            weights = np.array([ 0.1064484268, 0.0798886046, 0.0355538949, 0.0468614586,
                                 0.0533353277, 0.0544414595, 0.6914816499, 0.2314107567,
                                 0.0438039824, -0.0274356268, 0.0153638450, -0.0019912920,
                                 -0.0067333616])
            bias =  0.0023903439
            new_data = np.array([self.count_centre(state.local_board_status, player),
                                 self.count_corners(state.local_board_status, player),
                                 self.count_sides(state.local_board_status, player),
                                 sum(self.count_centre(each_local_board, player) for each_local_board in state.board.reshape(9, 3, 3)),
                                 sum(self.count_corners(each_local_board, player) for each_local_board in state.board.reshape(9, 3, 3)),
                                 sum(self.count_sides(each_local_board, player) for each_local_board in state.board.reshape(9, 3, 3)),
                                 self.global_three_in_a_row(state, player),
                                 self.x_in_a_row(state.local_board_status, player, 2),
                                 self.x_in_a_row(state.local_board_status, player, 1),
                                 self.local_three_in_a_row(state.local_board_status, player),
                                 sum(self.x_in_a_row(each_local_board, player, 2) for each_local_board in state.board.reshape(9, 3, 3)),
                                 sum(self.x_in_a_row(each_local_board, player, 1) for each_local_board in state.board.reshape(9, 3, 3)),
                                 self.has_free_movement(state)
                                 ]).reshape(1, -1)
            prediction = np.dot(new_data, weights) + bias
            return prediction
