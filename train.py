from utils import State, Action, load_data
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
from torch import tensor
import time

torch.set_printoptions(
    threshold=torch.inf,
    linewidth=200,
    sci_mode=False,
    precision=10
)

np.set_printoptions(
    threshold=np.inf,
    linewidth=200,
    suppress=True,
    precision=10
)

class StudentAgent:
    def __init__(self):
        self.states = None
        self.values = None
        self.global_boards = None
        self.local_boards = None 

    def test(self):
        states = []
        values = []

        data = load_data()
        for state, value in data:
            states.append(state)
            values.append(value)
            
        self.states = np.array(states)
        self.values = np.array(values)
        self.global_boards = np.array([state.local_board_status for state in states])
        self.local_boards = np.array([state.board.reshape(9, 3, 3) for state in states])

    def train_model(self):
        features = self.get_features()   
        labels = self.values
        model = LinearRegression()
        model.fit(features, labels)

        weights = model.coef_
        bias = model.intercept_
        print("weights = ")
        for w in weights:
            print(f"    {w:.10f},")
        print("bias = ", bias)

    def get_features(self):
        features = []

        # Won game
        won_game = np.array([self.won_game(state, 1) for state in self.states])

        # Global board
        global_board = np.array([board.flatten() for board in self.global_boards])
        global_board = np.where(global_board == 2, -1, global_board)

        # Whole board
        whole_board = np.array([state.board.flatten() for state in self.states])
        whole_board = np.where(whole_board == 2, -1, whole_board)

        # gloal two in a row (Player 1)
        global_two_in_a_row = np.array([self.x_in_a_row(board, 1, 2) for board in self.global_boards])

        # gloal two in a row (Player 2)
        global_two_in_a_row_2 = -1 * np.array([self.x_in_a_row(board, 2, 2) for board in self.global_boards])

        # global one in a row (Player 1)
        global_one_in_a_row = np.array([self.x_in_a_row(board, 1, 1) for board in self.global_boards])

        # global one in a row (Player 2)
        global_one_in_a_row_2 = -1 * np.array([self.x_in_a_row(board, 2, 1) for board in self.global_boards])

        # indiv local two in a row (Player 1)
        indiv_local_two_in_a_row = np.array([[self.x_in_a_row(each_local_board, 1, 2)
                                              for each_local_board in sample]
                                             for sample in self.local_boards
                                             ])

        # indiv local two in a row (Player 2)
        indiv_local_two_in_a_row_2 = -1 * np.array([[self.x_in_a_row(each_local_board, 2, 2)
                                              for each_local_board in sample]
                                             for sample in self.local_boards
                                             ])
        
        # indiv local one in a row (Player 1)
        indiv_local_one_in_a_row = np.array([[self.x_in_a_row(each_local_board, 1, 1)
                                              for each_local_board in sample]
                                             for sample in self.local_boards
                                             ])

        # indiv local one in a row (Player 2)
        indiv_local_one_in_a_row_2 = -1 * np.array([[self.x_in_a_row(each_local_board, 2, 1)
                                              for each_local_board in sample]
                                             for sample in self.local_boards
                                             ])

        # global centre (Player 1)
        global_centre = np.array([board[1][1] == 1 for board in self.global_boards]).astype(int)

        # global centre (Player 2)
        global_centre_2 = -1 * np.array([board[1][1] == 2 for board in self.global_boards]).astype(int)

        # global corners (Player 1)
        global_corners = np.array([self.total_corners(board, 1) for board in self.global_boards])

        # global corners (Player 2)
        global_corners_2 = -1 * np.array([self.total_corners(board, 2) for board in self.global_boards])

        # global sides (Player 1)
        global_sides = np.array([self.total_sides(board, 1) for board in self.global_boards])

        # global sides (Player 2)
        global_sides_2 = -1 * np.array([self.total_sides(board, 2) for board in self.global_boards])

        # local centres (Player 1)
        local_centres = np.array([[each_local_board[1][1] == 1
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ]).astype(int)

        # local centres (Player 2)
        local_centres_2 = -1 * np.array([[each_local_board[1][1] == 2
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ]).astype(int)
        
        # local corners (Player 1)
        local_corners = np.array([[self.total_corners(each_local_board, 1)
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ])

        # local corners (Player 2)
        local_corners_2 = -1 * np.array([[self.total_corners(each_local_board, 2)
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ])

        # local sides (Player 1)
        local_sides = np.array([[self.total_sides(each_local_board, 1)
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ])

        # local sides (Player 2)
        local_sides_2 = -1 * np.array([[self.total_sides(each_local_board, 2)
                                   for each_local_board in sample]
                                  for sample in self.local_boards
                                  ])

        # Current turn
        current_turn = np.array([1 if state.fill_num == 1 else -1 for state in self.states])

        # Current turn and Has free movement
        has_free_movement = np.array([self.has_free_movement(state) for state in self.states])
        
        features = np.column_stack([
            won_game,
            global_board,
            whole_board,
            global_two_in_a_row,
            global_two_in_a_row_2,
            global_one_in_a_row,
            global_one_in_a_row_2,
            indiv_local_two_in_a_row,
            indiv_local_two_in_a_row_2,
            indiv_local_one_in_a_row,
            indiv_local_one_in_a_row_2,
            global_centre,
            global_centre_2,
            global_corners,
            global_corners_2,
            global_sides,
            global_sides_2,
            local_centres,
            local_centres_2,
            local_corners,
            local_corners_2,
            local_sides,
            local_sides_2,
            current_turn,
            has_free_movement
            ])
        
        print("features shape = ", features.shape)
        return features
    
    def won_game(self, state: State, player: 1 | 2):
        if state.is_terminal():
            term_util = state.terminal_utility()
            if term_util == 1.0:
                if player == 1:
                    return 1
                else:
                    return -1
            elif term_util == 0.0:
                if player == 1:
                    return -1
                else:
                    return 1
            else:
                return 0
        else:
            return 0    

    def total_corners(self, board: np.array, player: 1 | 2):
        count = 0
        for row, col in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                if board[row][col] == player:
                    count += 1

        return count

    def total_sides(self, board: np.array, player: 1 | 2):
        count = 0
        for row, col in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                if board[row][col] == player:
                    count += 1

        return count

    def x_in_a_row(self, board: np.array, player: 1 | 2, x: int):
        count = 0

        for row in board:
            count_row_player = np.count_nonzero(row == player)
            count_row_zero = np.count_nonzero(row == 0)
            if count_row_player == x and count_row_zero == 3 - x:
                count += 1

        for col in np.transpose(board):
            count_col_player = np.count_nonzero(col == player)
            count_col_zero = np.count_nonzero(col == 0)
            if count_col_player == x and count_col_zero == 3 - x:
                count += 1

        board_diag = np.diag(board)
        count_diag_player = np.count_nonzero(board_diag == player)
        count_diag_zero = np.count_nonzero(board_diag == 0)
        if count_diag_player == x and count_diag_zero == 3 - x:
            count += 1

        other_board_diag = np.diag(np.fliplr(board))
        count_other_diag_player = np.count_nonzero(other_board_diag == player)
        count_other_diag_zero = np.count_nonzero(other_board_diag == 0)
        if count_other_diag_player == x and count_other_diag_zero == 3 - x:
            count += 1

        return count

    def has_free_movement(self, state: State):
        prev_action = state.prev_local_action
        if prev_action is None:
            return True
        row, col = prev_action
        return (state.local_board_status[row][col] != 0).astype(int)

    def choose_action(self, state: State):
        start_time = time.perf_counter()
        time_limit = 2.7 
        _, best_action = self.minimax(state, 1, -np.inf, np.inf, start_time, time_limit)
        depth = 3

        
        while True:
            print("depth = ", depth)
            if time.perf_counter() - start_time > time_limit:
                break
            _, action = self.minimax(state, depth, -np.inf, np.inf, start_time, time_limit)
            if time.perf_counter() - start_time < time_limit:
                best_action = action
            else:
                break
            depth += 1

        print("action = ", best_action)
        return best_action

    def minimax(self, state: State, max_depth: int, alpha: float, beta: float, start_time, time_limit):
        if state.is_terminal() or max_depth == 0 or time.perf_counter() - start_time > time_limit:
            return self.heuristic_func(state), None

        actions = state.get_all_valid_actions()

        if not actions:
            return self.heuristic_func(state), None

        best_action = None        
        is_maximizing = state.fill_num == 1
        best_value = -np.inf if is_maximizing else np.inf

        for action in actions:
            new_state = state.change_state(action)

            if is_maximizing:
                value = self.min_value(new_state, max_depth - 1, alpha, beta, start_time, time_limit)
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                value = self.max_value(new_state, max_depth - 1, alpha, beta, start_time, time_limit)
                if value < best_value:
                    best_value = value
                    best_action = action

            if is_maximizing:
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
            else:
                beta = min(beta, best_value)
                if best_value <= alpha:
                    break

        return best_value, best_action

    def max_value(self, state: State, depth: int, alpha: float, beta: float, start_time, time_limit):
        if state.is_terminal() or depth == 0 or time.perf_counter() - start_time > time_limit:
            return self.heuristic_func(state)

        v = -np.inf
        actions = state.get_all_valid_actions()

        for action in actions:
            next_state = state.change_state(action)
            v = max(v, self.min_value(next_state, depth - 1, alpha, beta, start_time, time_limit))
            alpha = max(alpha, v)
            if v >= beta:
                return v
        return v

    def min_value(self, state: State, depth: int, alpha: float, beta: float, start_time, time_limit):
        if state.is_terminal() or depth == 0 or time.perf_counter() - start_time > time_limit:
            return self.heuristic_func(state)

        v = np.inf
        actions = state.get_all_valid_actions()

        for action in actions:
            next_state = state.change_state(action)
            v = min(v, self.max_value(next_state, depth - 1, alpha, beta, start_time, time_limit))
            beta = min(beta, v)
            if v <= alpha:
                return v
        return v

    def heuristic_func(self, state: State):
        weights = np.array([1.2782281812,
    0.0212490228,
    -0.0141281772,
    0.0196679297,
    -0.0127409094,
    0.0165886022,
    -0.0095610855,
    0.0108306379,
    -0.0212394787,
    0.0112979031,
    0.0064539727,
    0.0056611200,
    -0.0024523596,
    0.0072563432,
    0.0010529861,
    0.0069018111,
    0.0025983673,
    0.0159924150,
    0.0117356422,
    0.0068201243,
    -0.0002239762,
    -0.0048257791,
    0.0092719374,
    -0.0156496039,
    0.0078597487,
    0.0122063280,
    0.0085271120,
    -0.0009443953,
    -0.0002409792,
    -0.0065020289,
    -0.0146579021,
    0.0114409453,
    -0.0246705526,
    -0.0027969426,
    0.0010118311,
    0.0001544512,
    0.0014721396,
    0.0052617556,
    0.0061529277,
    0.0042988543,
    0.0071548168,
    -0.0040160112,
    0.0107745285,
    0.0066399369,
    0.0128110697,
    0.0044385510,
    -0.0150552944,
    -0.0129453785,
    -0.0175019275,
    -0.0070280985,
    -0.0343393588,
    0.0023429604,
    -0.0089708072,
    -0.0037140215,
    -0.0042308716,
    -0.0021151385,
    0.0092645649,
    0.0062953516,
    0.0077175149,
    -0.0086123477,
    0.0025199970,
    0.0055602396,
    0.0118126177,
    0.0016699661,
    0.0068078496,
    0.0029290030,
    0.0082634148,
    0.0074046596,
    -0.0056772548,
    0.0155299010,
    -0.0050952752,
    -0.0027263689,
    -0.0008321017,
    0.0150071653,
    0.0043710138,
    -0.0050273839,
    0.0123376643,
    -0.0000573440,
    0.0153686497,
    0.0080651014,
    0.0064566820,
    0.0041593843,
    0.0152073459,
    0.0065332791,
    -0.0001748086,
    0.0126770245,
    0.0034472527,
    0.0073013672,
    0.0055405092,
    0.0088113887,
    0.0092009099,
    0.4062290447,
    0.3974393414,
    0.1357712144,
    0.1353278645,
    0.0714421340,
    0.0529786603,
    0.0883752504,
    0.0527236442,
    0.1182510384,
    0.0575203221,
    0.0793578757,
    0.0574091931,
    0.0660851613,
    0.0703374359,
    0.0425985938,
    0.0906579062,
    0.0447211256,
    0.1254703202,
    0.0526151238,
    0.0778266604,
    0.0446033218,
    0.0649365165,
    0.0261836020,
    0.0238190468,
    0.0330294877,
    0.0250514492,
    0.0477550025,
    0.0195854907,
    0.0281667529,
    0.0196971109,
    0.0254966656,
    0.0251039691,
    0.0236899537,
    0.0316750762,
    0.0175973977,
    0.0495565196,
    0.0219365588,
    0.0301369050,
    0.0160647511,
    0.0237549577,
    -0.1882851666,
    -0.1768386891,
    -0.1639644706,
    -0.1558060982,
    -0.0693032407,
    -0.0893225902,
    0.0123780043,
    -0.0068831229,
    -0.0176135361,
    0.0107041185,
    -0.0083551944,
    -0.0057470162,
    -0.0092790125,
    0.0067069295,
    0.0094850095,
    -0.0113250182,
    -0.0087664811,
    -0.0070570165,
    -0.0147201297,
    -0.0259841644,
    -0.0028653315,
    0.0036017578,
    -0.0067642735,
    -0.0060377568,
    0.0073083410,
    0.0051855720,
    -0.0060302496,
    0.0123188869,
    -0.0208103816,
    0.0118548551,
    0.0088011539,
    0.0152076524,
    0.0127735894,
    0.0110272815,
    0.0080707058,
    -0.0063846609,
    0.0083202110,
    -0.0249485192,
    -0.0004444362,
    0.0003427337,
    0.0069966148,
    0.0170003669,
    0.0138254900,
    0.0055385788,
    -0.0035916782,
    0.0185384232,
    -0.0205262174,
    0.0151525111,
    0.0018374288,
    0.0181490180,
    0.0141347658,
    0.0219861993,
    0.0198962432,
    0.0058881032,
    0.0183549195,
    -0.0008183207,
    0.0161621834,
    0.0212997659,
    0.0203849918,
    0.0211882938,
    0.2182744073,
    -0.0037465741])
        bias = -0.003847962382013741
        ref_state = state
                
        player_data = np.array([
            # Won game
            self.won_game(ref_state, 1),
            # Global board
            * np.where(ref_state.local_board_status.flatten() == 2, -1, ref_state.local_board_status.flatten()),
            # Whole board
            * np.where(ref_state.board.flatten() == 2, -1, ref_state.board.flatten()),
            # gloal two in a row (Player 1)
            self.x_in_a_row(ref_state.local_board_status, 1, 2),
            # gloal two in a row (Player 2)
            -1 * self.x_in_a_row(ref_state.local_board_status, 2, 2),
            # global one in a row (Player 1)
            self.x_in_a_row(ref_state.local_board_status, 1, 1),
            # global one in a row (Player 2)
            -1 * self.x_in_a_row(ref_state.local_board_status, 2, 1),
            # indiv local two in a row (Player 1)
            * [self.x_in_a_row(each_local_board, 1, 2) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # indiv local two in a row (Player 2)
            * [-1 *self.x_in_a_row(each_local_board, 2, 2) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # indiv local one in a row (Player 1)
            * [self.x_in_a_row(each_local_board, 1, 1) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # indiv local one in a row (Player 2)
            * [-1 * self.x_in_a_row(each_local_board, 2, 1) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # global centre (Player 1)
            (ref_state.local_board_status[1][1] == 1).astype(int),
            # global centre (Player 2)
            -1 * (ref_state.local_board_status[1][1] == 2).astype(int),
            # global corners (Player 1)
            self.total_corners(ref_state.local_board_status, 1),
            # global corners (Player 2)
            -1 * self.total_corners(ref_state.local_board_status, 2),
            # global sides (Player 1)
            self.total_sides(ref_state.local_board_status, 1),
            # global sides (Player 2)
            -1 * self.total_sides(ref_state.local_board_status, 2),
            # local centres (Player 1)
            * [int(each_local_board[1][1] == 1) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # local centres (Player 2)
            * [-1 * int(each_local_board[1][1] == 2) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # local corners (Player 1)
            * [self.total_corners(each_local_board, 1) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # local corners (Player 2)
            * [-1 * self.total_corners(each_local_board, 2) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # local sides (Player 1)
            * [self.total_sides(each_local_board, 1) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            # local sides (Player 2)
            * [-1 * self.total_sides(each_local_board, 2) for each_local_board in ref_state.board.reshape(9, 3, 3)],
            1 if ref_state.fill_num == 1 else -1,
            self.has_free_movement(ref_state)
            ]).flatten()
        
        heuristic = float(np.dot(weights, player_data) + bias)
        return heuristic
