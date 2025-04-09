import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
from app_test import Othello, create_graph, GNNQLearning, GNNMonteCarlo, minimax, evaluate_board, MCTS
from collections import deque
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Reward function (White's perspective)
def compute_reward(game, prev_game=None):
    if game.is_game_over():
        black_count, white_count = game.get_counts()
        return 50 * (white_count - black_count)  # White maximizes
    reward = evaluate_board(game)
    if prev_game:
        reward -= evaluate_board(prev_game)
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for r, c in corners:
        if game.board[r, c] == 1:  # White
            reward += 20
    return reward

# Training function for GNN + Q-Learning with self-play
def train_gnn_qlearning(model, target_model, optimizer, episodes=2000):
    game = Othello()
    replay_buffer = deque(maxlen=50000)
    update_target_every = 100
    
    for episode in range(episodes):
        game.reset()
        epsilon = max(0.01, 1.0 - episode / 1000)
        
        while not game.is_game_over():
            graph_data = create_graph(game.board).to(device)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
            # Self-play move selection
            if random.random() < epsilon:
                row, col = valid_moves[np.random.randint(len(valid_moves))]
            else:
                model.eval()
                with torch.no_grad():
                    q_values = model(graph_data).squeeze()
                valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
                best_move_idx = np.argmax(valid_q_values)
                row, col = valid_moves[best_move_idx]
            
            prev_game = game.copy()
            game.make_move(row, col)
            reward = compute_reward(game, prev_game) if game.current_player == 2 else -compute_reward(game, prev_game)  # White's perspective, negated for Black
            next_game = game.copy()
            
            replay_buffer.append((prev_game.board.copy(), row, col, reward, next_game.board.copy(), next_game.is_game_over()))
            
            # Train on batch
            if len(replay_buffer) >= 128:
                batch = random.sample(replay_buffer, 128)
                boards, rows, cols, rewards, next_boards, dones = zip(*batch)
                graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in boards])
                next_graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in next_boards])
                
                model.train()
                optimizer.zero_grad()
                q_preds = model(graph_batch).squeeze()
                with torch.no_grad():
                    next_q_values = target_model(next_graph_batch).squeeze()
                    next_q_max = torch.max(next_q_values.view(128, -1), dim=1)[0]
                
                indices = torch.tensor([i * 64 + r * 8 + c for i, (r, c) in enumerate(zip(rows, cols))], dtype=torch.long).to(device)
                q_preds_selected = q_preds[indices]
                q_targets = torch.tensor(rewards, dtype=torch.float).to(device) + \
                            0.99 * next_q_max * (1 - torch.tensor(dones, dtype=torch.float).to(device))
                
                loss = F.mse_loss(q_preds_selected, q_targets)
                loss.backward()
                optimizer.step()
        
        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        
        if episode % 100 == 0:
            print(f"Q-Learning Episode {episode} completed")
    
    torch.save(model.state_dict(), 'gnn_qlearning.pth')
    print("Saved GNN + Q-Learning model")

# Training function for GNN + MCTS with self-play
def train_gnn_mcts(model, optimizer, episodes=2000, simulations=100):
    game = Othello()
    replay_buffer = deque(maxlen=50000)
    
    for episode in range(episodes):
        game.reset()
        mcts = MCTS(model, simulations=simulations)
        episode_data = []  # (board, move, policy_target, value_target)
        
        while not game.is_game_over():
            board_tuple = tuple(game.board.flatten())
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
            # Self-play with MCTS
            for _ in range(simulations):
                mcts.simulate(game.copy())
            
            total_visits = sum(mcts.N.get((board_tuple, (r, c)), 0) for r, c in valid_moves)
            policy_target = torch.zeros(64, dtype=torch.float).to(device)
            for r, c in valid_moves:
                policy_target[r * 8 + c] = mcts.N.get((board_tuple, (r, c)), 0) / total_visits if total_visits > 0 else 1 / len(valid_moves)
            
            move_counts = [(mcts.N.get((board_tuple, (r, c)), 0), (r, c)) for r, c in valid_moves]
            row, col = max(move_counts, key=lambda x: x[0])[1]
            game.make_move(row, col)
            
            # Store experience (White's perspective)
            value_target = 1.0 if game.is_game_over() and game.get_counts()[1] > game.get_counts()[0] else None
            episode_data.append((game.board.copy(), (row, col), policy_target, value_target))
            
            if game.is_game_over():
                black_count, white_count = game.get_counts()
                final_value = 1.0 if white_count > black_count else -1.0 if black_count > white_count else 0.0
                for i, (board, move, pt, vt) in enumerate(episode_data):
                    # Adjust value based on player's turn at move time
                    player_at_move = 1 if (len(episode_data) - i) % 2 == 0 else 2  # White moves on even steps from end
                    adjusted_value = final_value if player_at_move == 1 else -final_value
                    episode_data[i] = (board, move, pt, adjusted_value if vt is None else vt)
                replay_buffer.extend(episode_data)
        
        # Train on batch
        if len(replay_buffer) >= 128:
            batch = random.sample(replay_buffer, 128)
            boards, _, policy_targets, value_targets = zip(*batch)
            graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in boards])
            
            model.train()
            optimizer.zero_grad()
            value_preds, policy_preds = model(graph_batch)
            value_loss = F.mse_loss(value_preds.squeeze(), torch.tensor(value_targets, dtype=torch.float).to(device))
            policy_loss = F.cross_entropy(policy_preds.view(128, -1), torch.stack(policy_targets))
            loss = value_loss + policy_loss
            
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"MCTS Episode {episode} completed")

    torch.save(model.state_dict(), 'gnn_montecarlo.pth')
    print("Saved GNN + MCTS model")

# Test GNN vs Minimax (GNN as White)
def test_vs_minimax(gnn_model, num_games=20):
    wins = 0
    for _ in range(num_games):
        game = Othello()
        while not game.is_game_over():
            if game.current_player == 2:  # Black (Minimax)
                move = minimax(game, 5, True)
                if move:
                    game.make_move(*move)
            else:  # White (GNN)
                graph_data = create_graph(game.board).to(device)
                gnn_model.eval()
                with torch.no_grad():
                    if isinstance(gnn_model, GNNMonteCarlo):
                        _, policy = gnn_model(graph_data)
                        valid_moves = game.get_valid_moves()
                        valid_q_values = [policy[row * 8 + col].item() for row, col in valid_moves]
                    else:
                        q_values = gnn_model(graph_data).squeeze()
                        valid_moves = game.get_valid_moves()
                        valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
                best_move_idx = np.argmax(valid_q_values)
                game.make_move(*valid_moves[best_move_idx])
        black_count, white_count = game.get_counts()
        if white_count > black_count:
            wins += 1
    return wins / num_games

# Test GNN vs Random (GNN as White)
def test_vs_random(gnn_model, num_games=20):
    wins = 0
    for _ in range(num_games):
        game = Othello()
        while not game.is_game_over():
            if game.current_player == 2:  # Black (Random)
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    game.make_move(*valid_moves[np.random.randint(len(valid_moves))])
            else:  # White (GNN)
                graph_data = create_graph(game.board).to(device)
                gnn_model.eval()
                with torch.no_grad():
                    if isinstance(gnn_model, GNNMonteCarlo):
                        _, policy = gnn_model(graph_data)
                        valid_moves = game.get_valid_moves()
                        valid_q_values = [policy[row * 8 + col].item() for row, col in valid_moves]
                    else:
                        q_values = gnn_model(graph_data).squeeze()
                        valid_moves = game.get_valid_moves()
                        valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
                best_move_idx = np.argmax(valid_q_values)
                game.make_move(*valid_moves[best_move_idx])
        black_count, white_count = game.get_counts()
        if white_count > black_count:
            wins += 1
    return wins / num_games

if __name__ == "__main__":
    # Train GNN + Q-Learning
    gnn_qlearning = GNNQLearning().to(device)
    target_qlearning = GNNQLearning().to(device)
    target_qlearning.load_state_dict(gnn_qlearning.state_dict())
    optimizer_qlearning = torch.optim.Adam(gnn_qlearning.parameters(), lr=0.0005)
    train_gnn_qlearning(gnn_qlearning, target_qlearning, optimizer_qlearning)
    qlearning_minimax_win_rate = test_vs_minimax(gnn_qlearning)
    qlearning_random_win_rate = test_vs_random(gnn_qlearning)
    print(f"GNN + Q-Learning (White) win rate vs Minimax: {qlearning_minimax_win_rate:.2%}")
    print(f"GNN + Q-Learning (White) win rate vs Random: {qlearning_random_win_rate:.2%}")

    # Train GNN + MCTS
    gnn_montecarlo = GNNMonteCarlo().to(device)
    optimizer_montecarlo = torch.optim.Adam(gnn_montecarlo.parameters(), lr=0.0005)
    train_gnn_mcts(gnn_montecarlo, optimizer_montecarlo)
    mcts_minimax_win_rate = test_vs_minimax(gnn_montecarlo)
    mcts_random_win_rate = test_vs_random(gnn_montecarlo)
    print(f"GNN + MCTS (White) win rate vs Minimax: {mcts_minimax_win_rate:.2%}")
    print(f"GNN + MCTS (White) win rate vs Random: {mcts_random_win_rate:.2%}")