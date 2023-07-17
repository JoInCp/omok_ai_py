# 알파베타 vs 몬테카를로
import random

# 오목 게임 상태 클래스
class GameState:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1

    # 현재 가능한 모든 수의 좌표를 반환
    def get_possible_moves(self):
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    # 현재 상태에서 해당 좌표에 돌을 놓음
    def make_move(self, move):
        x, y = move
        self.board[x][y] = self.current_player
        self.current_player = 3 - self.current_player  # 플레이어 번갈아가며 두도록 변경

    # 현재 게임 상태를 복사하여 반환
    def copy(self):
        copied_state = GameState(self.board_size)
        copied_state.board = [row[:] for row in self.board]
        copied_state.current_player = self.current_player
        return copied_state

    # 현재 게임 상태에서 승리 조건을 만족하는지 확인
    def check_win(self):
        # 가로 방향 체크
        for i in range(self.board_size):
            for j in range(self.board_size - 4):
                if (
                    self.board[i][j]
                    and self.board[i][j] == self.board[i][j + 1]
                    and self.board[i][j] == self.board[i][j + 2]
                    and self.board[i][j] == self.board[i][j + 3]
                    and self.board[i][j] == self.board[i][j + 4]
                ):
                    return self.board[i][j]

        # 세로 방향 체크
        for i in range(self.board_size - 4):
            for j in range(self.board_size):
                if (
                    self.board[i][j]
                    and self.board[i][j] == self.board[i + 1][j]
                    and self.board[i][j] == self.board[i + 2][j]
                    and self.board[i][j] == self.board[i + 3][j]
                    and self.board[i][j] == self.board[i + 4][j]
                ):
                    return self.board[i][j]

        # 대각선 방향 체크
        for i in range(self.board_size - 4):
            for j in range(self.board_size - 4):
                if (
                    self.board[i][j]
                    and self.board[i][j] == self.board[i + 1][j + 1]
                    and self.board[i][j] == self.board[i + 2][j + 2]
                    and self.board[i][j] == self.board[i + 3][j + 3]
                    and self.board[i][j] == self.board[i + 4][j + 4]
                ):
                    return self.board[i][j]

                if (
                    self.board[i + 4][j]
                    and self.board[i + 4][j] == self.board[i + 3][j + 1]
                    and self.board[i + 4][j] == self.board[i + 2][j + 2]
                    and self.board[i + 4][j] == self.board[i + 1][j + 3]
                    and self.board[i + 4][j] == self.board[i][j + 4]
                ):
                    return self.board[i + 4][j]

        return 0

# 알파베타 가지치기를 사용하는 AI 클래스
class AlphaBetaAI:
    def __init__(self, board_size, max_depth):
        self.board_size = board_size
        self.max_depth = max_depth

    # 알파베타 가지치기 알고리즘을 사용하여 최선의 수를 찾음
    def get_best_move(self, state):
        best_score = float('-inf')
        best_move = None

        for move in state.get_possible_moves():
            new_state = state.copy()
            new_state.make_move(move)
            score = self.alpha_beta_pruning(new_state, self.max_depth, float('-inf'), float('inf'), False)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # 알파베타 가지치기 알고리즘
    def alpha_beta_pruning(self, state, depth, alpha, beta, is_maximizing):
        if depth == 0 or state.check_win() != 0:
            return self.evaluate(state)

        if is_maximizing:
            max_eval = float('-inf')
            for move in state.get_possible_moves():
                new_state = state.copy()
                new_state.make_move(move)
                eval = self.alpha_beta_pruning(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in state.get_possible_moves():
                new_state = state.copy()
                new_state.make_move(move)
                eval = self.alpha_beta_pruning(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    # 게임 상태를 평가하여 점수를 반환
    def evaluate(self, state):
        score = 0
        player = state.current_player

        # 가로 방향 체크
        for i in range(state.board_size):
            for j in range(state.board_size - 4):
                window = state.board[i][j:j+5]
                score += self.evaluate_window(window, player)

        # 세로 방향 체크
        for i in range(state.board_size - 4):
            for j in range(state.board_size):
                window = [state.board[k][j] for k in range(i, i+5)]
                score += self.evaluate_window(window, player)

        # 대각선 방향 체크 (왼쪽 위에서 오른쪽 아래로)
        for i in range(state.board_size - 4):
            for j in range(state.board_size - 4):
                window = [state.board[i+k][j+k] for k in range(5)]
                score += self.evaluate_window(window, player)

        # 대각선 방향 체크 (왼쪽 아래에서 오른쪽 위로)
        for i in range(state.board_size - 4):
            for j in range(4, state.board_size):
                window = [state.board[i+k][j-k] for k in range(5)]
                score += self.evaluate_window(window, player)

        return score

    # 윈도우 내에서 특정 플레이어의 점수를 계산
    def evaluate_window(self, window, player):
        score = 0
        opponent = 3 - player

        if window.count(player) == 5:
            score += 100
        elif window.count(player) == 4 and window.count(0) == 1:
            score += 10
        elif window.count(player) == 3 and window.count(0) == 2:
            score += 5

        if window.count(opponent) == 4 and window.count(0) == 1:
            score -= 20

        return score

# 몬테 카를로 트리 탐색을 사용하는 AI 클래스
class MonteCarloAI:
    def __init__(self, board_size, simulations):
        self.board_size = board_size
        self.simulations = simulations

    # 몬테 카를로 트리 탐색 알고리즘을 사용하여 최선의 수를 찾음
    def get_best_move(self, state):
        scores = dict()

        for move in state.get_possible_moves():
            scores[move] = 0

            for _ in range(self.simulations):
                new_state = state.copy()
                new_state.make_move(move)
                scores[move] += self.simulate(new_state)

        best_move = max(scores, key=scores.get)
        return best_move

    # 무작위 시뮬레이션을 통해 게임을 진행하고 승리한 횟수를 반환
    def simulate(self, state):
        while state.check_win() == 0:
            possible_moves = state.get_possible_moves()
            move = random.choice(possible_moves)
            state.make_move(move)
        winner = state.check_win()
        return 1 if winner == 1 else 0

# AI 간의 대결을 수행하고 승률을 출력
def play_games(num_games):
    board_size = 15
    max_depth = 4
    simulations = 100

    alpha_beta_total_wins = 0
    monte_carlo_total_wins = 0
    draws = 0

    for _ in range(num_games):
        alpha_beta_wins = 0
        monte_carlo_wins = 0
        draw_count = 0

        for _ in range(100):
            state = GameState(board_size)
            alpha_beta_ai = AlphaBetaAI(board_size, max_depth)
            monte_carlo_ai = MonteCarloAI(board_size, simulations)

            while state.check_win() == 0:
                if state.current_player == 1:
                    move = alpha_beta_ai.get_best_move(state)
                else:
                    move = monte_carlo_ai.get_best_move(state)
                state.make_move(move)

            winner = state.check_win()
            if winner == 1:
                alpha_beta_wins += 1
            elif winner == 2:
                monte_carlo_wins += 1
            else:
                draw_count += 1

        alpha_beta_total_wins += alpha_beta_wins
        monte_carlo_total_wins += monte_carlo_wins
        draws += draw_count

        print(f"Game: {_ + 1}")
        print("AlphaBeta AI Wins:", alpha_beta_wins)
        print("Monte Carlo AI Wins:", monte_carlo_wins)
        print("Draws:", draw_count)
        print()

    print("=== Final Results ===")
    print("AlphaBeta AI Wins:", alpha_beta_total_wins)
    print("Monte Carlo AI Wins:", monte_carlo_total_wins)
    print("Draws:", draws)

    print("=== Final Results ===")
    print("AlphaBeta AI Win Rate:", alpha_beta_total_wins / (num_games * 100) * 100, "%")
    print("Monte Carlo AI Win Rate:", monte_carlo_total_wins / (num_games * 100) * 100, "%")
    print("Draw Rate:", draws / (num_games * 100) * 100, "%")

# 게임 횟수 입력받고 대결 수행
num_games = int(input("게임 횟수를 입력하세요: "))
play_games(num_games)
