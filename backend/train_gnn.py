import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
from app import Othello, create_graph, GNNQLearning, GNNMonteCarlo, minimax, evaluate_board
from collections import deque
import random
import copy

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Enhanced reward function
def compute_reward(game, prev_game=None):
    if game.is_game_over():
        black_count, white_count = game.get_counts()
        return 50 * (black_count - white_count) if game.current_player == 2 else 50 * (white_count - black_count)
    reward = evaluate_board(game)  # Base on Minimax heuristic
    if prev_game:
        reward -= evaluate_board(prev_game)  # Differential reward
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for r, c in corners:
        if game.board[r, c] == game.current_player:
            reward += 20  # Bonus for corners
    return reward

# Training function for GNN + Double Q-Learning (DDQN)
def train_gnn_qlearning(model, target_model, optimizer, episodes=2000):
    game = Othello()
    replay_buffer = deque(maxlen=50000)  # Larger buffer
    update_target_every = 100  # Sync target network
    
    for episode in range(episodes):
        game.reset()
        epsilon = max(0.01, 1.0 - episode / 1000)  # Slower decay
        
        while not game.is_game_over():
            graph_data = create_graph(game.board).to(device)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
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
            reward = compute_reward(game, prev_game)
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
                    next_q_values = target_model(next_graph_batch).squeeze()  # Use target network
                    next_q_max = torch.max(next_q_values.view(128, -1), dim=1)[0]
                
                indices = torch.tensor([i * 64 + r * 8 + c for i, (r, c) in enumerate(zip(rows, cols))], dtype=torch.long).to(device)
                q_preds_selected = q_preds[indices]
                q_targets = torch.tensor(rewards, dtype=torch.float).to(device) + \
                            0.99 * next_q_max * (1 - torch.tensor(dones, dtype=torch.float).to(device))
                
                loss = F.mse_loss(q_preds_selected, q_targets)
                loss.backward()
                optimizer.step()
        
        # Update target network
        if episode % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        
        if episode % 100 == 0:
            print(f"Q-Learning Episode {episode} completed")
    
    torch.save(model.state_dict(), 'gnn_qlearning.pth')
    print("Saved GNN + Q-Learning model")

# Training function for GNN + Monte Carlo with PPO-like policy
def train_gnn_montecarlo(model, optimizer, episodes=2000, simulations=20):
    game = Othello()
    replay_buffer = deque(maxlen=50000)
    epsilon = 0.1
    
    for episode in range(episodes):
        game.reset()
        episode_data = []  # (board, action, reward, log_prob)
        
        while not game.is_game_over():
            graph_data = create_graph(game.board).to(device)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
            model.eval()
            with torch.no_grad():
                q_values = model(graph_data).squeeze()
                valid_q_values = torch.tensor([q_values[row * 8 + col].item() for row, col in valid_moves], dtype=torch.float).to(device)
                probs = F.softmax(valid_q_values, dim=0)
                move_idx = torch.multinomial(probs, 1).item()
                row, col = valid_moves[move_idx]
                log_prob = torch.log(probs[move_idx])
            
            prev_game = game.copy()
            game.make_move(row, col)
            reward = compute_reward(game, prev_game)
            episode_data.append((prev_game.board.copy(), (row, col), reward, log_prob))
            
            if game.is_game_over():
                black_count, white_count = game.get_counts()
                final_reward = 50 * (black_count - white_count) if game.current_player == 2 else 50 * (white_count - black_count)
                for i, (board, _, r, _) in enumerate(episode_data):
                    episode_data[i] = (board, episode_data[i][1], r + final_reward * (0.99 ** (len(episode_data) - i - 1)), episode_data[i][3])
                replay_buffer.extend(episode_data)
        
        # Train on batch with PPO-like update
        if len(replay_buffer) >= 128:
            batch = random.sample(replay_buffer, 128)
            boards, actions, rewards, log_probs = zip(*batch)
            graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in boards])
            
            model.train()
            optimizer.zero_grad()
            q_values = model(graph_batch).squeeze()
            indices = torch.tensor([i * 64 + r * 8 + c for i, (r, c) in enumerate(actions)], dtype=torch.long).to(device)
            new_q_values = q_values[indices]
            new_log_probs = torch.log(F.softmax(q_values.view(128, -1), dim=1)[range(128), indices % 64])
            
            advantages = torch.tensor(rewards, dtype=torch.float).to(device) - new_q_values.detach()
            ratio = torch.exp(new_log_probs - torch.tensor(log_probs, dtype=torch.float).to(device))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages  # PPO clipping
            loss = -torch.min(surr1, surr2).mean()  # Policy loss
            
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"Monte Carlo Episode {episode} completed")

    torch.save(model.state_dict(), 'gnn_montecarlo.pth')
    print("Saved GNN + Monte Carlo model")

# Test GNN vs Minimax
def test_vs_minimax(gnn_model, num_games=20):
    wins = 0
    for _ in range(num_games):
        game = Othello()
        while not game.is_game_over():
            if game.current_player == 2:  # GNN as Black
                graph_data = create_graph(game.board).to(device)
                gnn_model.eval()
                with torch.no_grad():
                    q_values = gnn_model(graph_data).squeeze()
                valid_moves = game.get_valid_moves()
                valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
                best_move_idx = np.argmax(valid_q_values)
                game.make_move(*valid_moves[best_move_idx])
            else:  # Minimax as White
                move = minimax(game, 5, True)
                if move:
                    game.make_move(*move)
        black_count, white_count = game.get_counts()
        if black_count > white_count:
            wins += 1
    return wins / num_games

# Test GNN vs Random
def test_vs_random(gnn_model, num_games=20):
    wins = 0
    for _ in range(num_games):
        game = Othello()
        while not game.is_game_over():
            if game.current_player == 2:  # GNN as Black
                graph_data = create_graph(game.board).to(device)
                gnn_model.eval()
                with torch.no_grad():
                    q_values = gnn_model(graph_data).squeeze()
                valid_moves = game.get_valid_moves()
                valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
                best_move_idx = np.argmax(valid_q_values)
                game.make_move(*valid_moves[best_move_idx])
            else:  # Random as White
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    game.make_move(*valid_moves[np.random.randint(len(valid_moves))])
        black_count, white_count = game.get_counts()
        if black_count > white_count:
            wins += 1
    return wins / num_games

if __name__ == "__main__":
    # Train GNN + Q-Learning
    gnn_qlearning = GNNQLearning().to(device)
    target_qlearning = GNNQLearning().to(device)
    target_qlearning.load_state_dict(gnn_qlearning.state_dict())
    optimizer_qlearning = torch.optim.Adam(gnn_qlearning.parameters(), lr=0.0005)  # Lower LR for stability
    train_gnn_qlearning(gnn_qlearning, target_qlearning, optimizer_qlearning)
    qlearning_minimax_win_rate = test_vs_minimax(gnn_qlearning)
    qlearning_random_win_rate = test_vs_random(gnn_qlearning)
    print(f"GNN + Q-Learning win rate vs Minimax: {qlearning_minimax_win_rate:.2%}")
    print(f"GNN + Q-Learning win rate vs Random: {qlearning_random_win_rate:.2%}")

    # Train GNN + Monte Carlo
    gnn_montecarlo = GNNMonteCarlo().to(device)
    optimizer_montecarlo = torch.optim.Adam(gnn_montecarlo.parameters(), lr=0.0005)
    train_gnn_montecarlo(gnn_montecarlo, optimizer_montecarlo)
    montecarlo_minimax_win_rate = test_vs_minimax(gnn_montecarlo)
    montecarlo_random_win_rate = test_vs_random(gnn_montecarlo)
    print(f"GNN + Monte Carlo win rate vs Minimax: {montecarlo_minimax_win_rate:.2%}")
    print(f"GNN + Monte Carlo win rate vs Random: {montecarlo_random_win_rate:.2%}")