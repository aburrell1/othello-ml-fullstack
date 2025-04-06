from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Othello board as a graph
def create_graph(board):
    num_nodes = 64
    x = torch.tensor(board.flatten(), dtype=torch.float).view(-1, 1)  # 0=empty, 1=white, 2=black
    edge_index = []
    for i in range(8):
        for j in range(8):
            node = i * 8 + j
            if i > 0: edge_index.append([node, node - 8])  # Up
            if i < 7: edge_index.append([node, node + 8])  # Down
            if j > 0: edge_index.append([node, node - 1])  # Left
            if j < 7: edge_index.append([node, node + 1])  # Right
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# GNN Models
class GNNQLearning(torch.nn.Module):
    def __init__(self):
        super(GNNQLearning, self).__init__()
        self.conv1 = GCNConv(1, 32) 
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

class GNNMonteCarlo(torch.nn.Module):
    def __init__(self):
        super(GNNMonteCarlo, self).__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)
        self.conv4 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

# Othello Game Logic
class Othello:
    def __init__(self):
        self.reset()
        self.move_history = []

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3] = self.board[4, 4] = 1  # White
        self.board[3, 4] = self.board[4, 3] = 2  # Black
        self.current_player = 2  # Black starts
        self.move_history = []

    def get_valid_moves(self):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(i, j, self.current_player):
                    moves.append([i, j])
        return moves

    def is_valid_move(self, row, col, player):
        if self.board[row, col] != 0:
            return False
        opponent = 3 - player
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
                found_opponent = True
                r, c = r + dr, c + dc
            if found_opponent and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                return True
        return False

    def make_move(self, row, col):
        if not self.is_valid_move(row, col, self.current_player):
            return False
        opponent = 3 - self.current_player
        flipped = []
        self.board[row, col] = self.current_player
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
                to_flip.append((r, c))
                r, c = r + dr, c + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == self.current_player:
                for fr, fc in to_flip:
                    self.board[fr, fc] = self.current_player
                    flipped.append((fr, fc))
        self.move_history.append((row, col, flipped, self.current_player))
        self.current_player = 3 - self.current_player
        self.skip_turn_if_needed()
        return True

    def undo_move(self):
        if not self.move_history:
            return False
        row, col, flipped, player = self.move_history.pop()
        self.board[row, col] = 0
        opponent = 3 - player
        for fr, fc in flipped:
            self.board[fr, fc] = opponent
        self.current_player = player
        return True

    def skip_turn_if_needed(self):
        current_moves = self.get_valid_moves()
        if not current_moves:
            self.current_player = 3 - self.current_player
            opponent_moves = self.get_valid_moves()
            if not opponent_moves:
                return False
            return True
        return False

    def is_game_over(self):
        current_moves = self.get_valid_moves()
        self.current_player = 3 - self.current_player
        opponent_moves = self.get_valid_moves()
        self.current_player = 3 - self.current_player
        return not current_moves and not opponent_moves or np.all(self.board != 0).item()

    def get_counts(self):
        black_count = np.sum(self.board == 2).item()
        white_count = np.sum(self.board == 1).item()
        return black_count, white_count

    def copy(self):
        new_game = Othello()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        return new_game

# Monte Carlo Simulation
def monte_carlo_simulation(game, move, simulations=100):
    wins = 0
    for _ in range(simulations):
        sim_game = game.copy()
        sim_game.make_move(move[0], move[1])
        while not sim_game.is_game_over():
            valid_moves = sim_game.get_valid_moves()
            if valid_moves:
                row, col = valid_moves[np.random.randint(len(valid_moves))]
                sim_game.make_move(row, col)
            else:
                sim_game.current_player = 3 - sim_game.current_player
        black_count, white_count = sim_game.get_counts()
        wins += 1 if (sim_game.current_player == 2 and black_count > white_count) or \
                     (sim_game.current_player == 1 and white_count > black_count) else 0
    return wins / simulations

# Minimax with Heuristic
def minimax(game, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
    if depth == 0 or game.is_game_over():
        return evaluate_board(game)
    
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return evaluate_board(game)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in valid_moves:
            new_game = game.copy()
            new_game.make_move(move[0], move[1])
            eval_score = minimax(new_game, depth - 1, False, alpha, beta)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval if depth != 5 else best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_game = game.copy()
            new_game.make_move(move[0], move[1])
            eval_score = minimax(new_game, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def evaluate_board(game):
    black_count, white_count = game.get_counts()
    coin_parity = 100 * (black_count - white_count) / (black_count + white_count + 1)
    current_mobility = len(game.get_valid_moves())
    game.current_player = 3 - game.current_player
    opponent_mobility = len(game.get_valid_moves())
    game.current_player = 3 - game.current_player
    mobility = 100 * (current_mobility - opponent_mobility) / (current_mobility + opponent_mobility + 1)
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    corner_score = 0
    for r, c in corners:
        if game.board[r, c] == game.current_player:
            corner_score += 25
        elif game.board[r, c] == (3 - game.current_player):
            corner_score -= 25
    return coin_parity + mobility + corner_score

# Initialize game and models
game = Othello()
gnn_qlearning = GNNQLearning()
gnn_montecarlo = GNNMonteCarlo()
optimizer_qlearning = torch.optim.Adam(gnn_qlearning.parameters(), lr=0.001)  # Lower lr for stability
optimizer_montecarlo = torch.optim.Adam(gnn_montecarlo.parameters(), lr=0.001)
ai_type = 'gnn_qlearning'

# Load trained models if they exist
if os.path.exists('gnn_qlearning.pth'):
    gnn_qlearning.load_state_dict(torch.load('gnn_qlearning.pth'))
    print("Loaded trained GNN + Q-Learning model")
if os.path.exists('gnn_montecarlo.pth'):
    gnn_montecarlo.load_state_dict(torch.load('gnn_montecarlo.pth'))
    print("Loaded trained GNN + Monte Carlo model")

# API endpoints
@app.route('/api/board', methods=['GET'])
def get_board():
    game.skip_turn_if_needed()
    game_over = game.is_game_over()
    black_count, white_count = game.get_counts()
    valid_moves = game.get_valid_moves()
    return jsonify({
        'board': game.board.tolist(),
        'currentPlayer': game.current_player,
        'gameOver': game_over,
        'blackCount': black_count,
        'whiteCount': white_count,
        'validMoves': valid_moves
    })

@app.route('/api/move', methods=['POST'])
def player_move():
    data = request.json
    row, col = data['row'], data['col']
    success = game.make_move(row, col)
    if success:
        game.skip_turn_if_needed()
    game_over = game.is_game_over()
    black_count, white_count = game.get_counts()
    valid_moves = game.get_valid_moves()
    return jsonify({
        'success': success,
        'board': game.board.tolist(),
        'currentPlayer': game.current_player,
        'gameOver': game_over,
        'blackCount': black_count,
        'whiteCount': white_count,
        'validMoves': valid_moves
    })

@app.route('/api/ai-move', methods=['POST'])
def ai_move():
    global ai_type
    if game.skip_turn_if_needed():
        black_count, white_count = game.get_counts()
        valid_moves = game.get_valid_moves()
        return jsonify({
            'success': False,
            'skipped': True,
            'message': f'Player {3 - game.current_player} has no moves, turn skipped',
            'board': game.board.tolist(),
            'currentPlayer': game.current_player,
            'gameOver': game.is_game_over(),
            'blackCount': black_count,
            'whiteCount': white_count,
            'validMoves': valid_moves
        })

    valid_moves = game.get_valid_moves()
    if not valid_moves:
        game.current_player = 3 - game.current_player
        black_count, white_count = game.get_counts()
        valid_moves = game.get_valid_moves()
        return jsonify({
            'success': False,
            'skipped': True,
            'message': 'AI has no valid moves, turn skipped',
            'board': game.board.tolist(),
            'currentPlayer': game.current_player,
            'gameOver': game.is_game_over(),
            'blackCount': black_count,
            'whiteCount': white_count,
            'validMoves': valid_moves
        })

    if ai_type == 'gnn_qlearning':
        graph_data = create_graph(game.board)
        gnn_qlearning.eval()
        with torch.no_grad():
            q_values = gnn_qlearning(graph_data).squeeze()
        
        valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        row, col = valid_moves[best_move_idx]
        
        game.make_move(row, col)
        
        # Improved Q-Learning update
        next_game = game.copy()
        next_valid_moves = next_game.get_valid_moves()
        reward = 1 if next_valid_moves else -1
        if next_game.is_game_over():
            black_count, white_count = next_game.get_counts()
            reward = -(10 * (black_count - white_count)) if black_count > white_count else (10 * (white_count - black_count))
        
        gnn_qlearning.train()
        optimizer_qlearning.zero_grad()
        q_pred = gnn_qlearning(graph_data)[row * 8 + col]
        next_q_values = gnn_qlearning(create_graph(next_game.board)).squeeze()
        q_target = reward + 0.9 * (max([next_q_values[m[0] * 8 + m[1]] for m in next_valid_moves], default=0) if next_valid_moves else 0)
        loss = F.mse_loss(q_pred, torch.tensor(q_target, dtype=torch.float))
        loss.backward()
        optimizer_qlearning.step()

    elif ai_type == 'gnn_montecarlo':
        graph_data = create_graph(game.board)
        gnn_montecarlo.eval()
        with torch.no_grad():
            q_values = gnn_montecarlo(graph_data).squeeze()
        
        valid_q_values = [(q_values[row * 8 + col].item() + monte_carlo_simulation(game, [row, col])) for row, col in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        row, col = valid_moves[best_move_idx]
        
        game.make_move(row, col)

    else:  # minimax
        move = minimax(game, 5, True)
        if move:
            row, col = move
            game.make_move(row, col)

    game_over = game.is_game_over()
    black_count, white_count = game.get_counts()
    valid_moves = game.get_valid_moves()
    return jsonify({
        'success': True,
        'row': row,
        'col': col,
        'board': game.board.tolist(),
        'currentPlayer': game.current_player,
        'gameOver': game_over,
        'blackCount': black_count,
        'whiteCount': white_count,
        'validMoves': valid_moves
    })

@app.route('/api/reset', methods=['POST'])
def reset_game():
    game.reset()
    black_count, white_count = game.get_counts()
    valid_moves = game.get_valid_moves()
    return jsonify({
        'board': game.board.tolist(),
        'currentPlayer': game.current_player,
        'gameOver': False,
        'blackCount': black_count,
        'whiteCount': white_count,
        'validMoves': valid_moves
    })

@app.route('/api/undo', methods=['POST'])
def undo_move():
    success = game.undo_move()
    if success:
        game.skip_turn_if_needed()
    game_over = game.is_game_over()
    black_count, white_count = game.get_counts()
    valid_moves = game.get_valid_moves()
    return jsonify({
        'success': success,
        'board': game.board.tolist(),
        'currentPlayer': game.current_player,
        'gameOver': game_over,
        'blackCount': black_count,
        'whiteCount': white_count,
        'validMoves': valid_moves
    })

@app.route('/api/set-ai', methods=['POST'])
def set_ai():
    global ai_type
    data = request.json
    ai_type = data['aiType']
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)