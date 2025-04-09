import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np
from app import Othello, create_graph, GNNQLearning, GNNMonteCarlo, minimax, evaluate_board 
from collections import deque
import random

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Training function for GNN + Q-Learning
def train_gnn_qlearning(model, optimizer, episodes=1000):
    game = Othello()
    replay_buffer = deque(maxlen=20000)
    
    for episode in range(episodes):
        game.reset()
        epsilon = max(0.05, 1.0 - episode / 500)  
        
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
            
            prev_board = game.board.copy()
            game.make_move(row, col)
            next_game = game.copy()
            next_valid_moves = next_game.get_valid_moves()
            reward = evaluate_board(game) - evaluate_board(next_game)  
            if next_game.is_game_over():
                black_count, white_count = next_game.get_counts()
                reward = -(20 * (black_count - white_count)) if black_count > white_count else 20 * (white_count - black_count)
            
            replay_buffer.append((prev_board, row, col, reward, next_game.board.copy()))
            
            # Train on batch
            if len(replay_buffer) >= 64:
                batch = random.sample(replay_buffer, 64)
                boards, rows, cols, rewards, next_boards = zip(*batch)
                graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in boards])
                next_graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in next_boards])
                
                model.train()
                optimizer.zero_grad()
                q_preds = model(graph_batch).squeeze()
                next_q_values = model(next_graph_batch).squeeze()
                
                indices = torch.tensor([i * 64 + r * 8 + c for i, (r, c) in enumerate(zip(rows, cols))], dtype=torch.long).to(device)
                q_preds_selected = q_preds[indices]
                next_q_max = torch.max(next_q_values.view(64, -1), dim=1)[0]  
                q_targets = torch.tensor(rewards, dtype=torch.float).to(device) + 0.95 * next_q_max
                
                loss = F.mse_loss(q_preds_selected, q_targets)
                loss.backward()
                optimizer.step()
        
        if episode % 100 == 0:
            print(f"Q-Learning Episode {episode} completed")
    
    torch.save(model.state_dict(), 'gnn_qlearning.pth')
    print("Saved GNN + Q-Learning model")

# Training function for GNN + Monte Carlo
def train_gnn_montecarlo(model, optimizer, episodes=1000, simulations=10):
    game = Othello()
    replay_buffer = deque(maxlen=20000)
    epsilon = 0.1
    
    for episode in range(episodes):
        game.reset()
        episode_moves = []
        
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
            
            episode_moves.append((game.board.copy(), row, col))
            game.make_move(row, col)
            
            if game.is_game_over():
                black_count, white_count = game.get_counts()
                final_reward = -(20 * (black_count - white_count)) if black_count > white_count else 20 * (white_count - black_count)
                
                for board, r, c in episode_moves:
                    sim_game = Othello()
                    sim_game.board = board.copy()
                    sim_game.current_player = game.current_player
                    mc_value = monte_carlo_simulation(sim_game, [r, c], simulations)
                    replay_buffer.append((board, r, c, mc_value + final_reward))
        
        if len(replay_buffer) >= 64:
            batch = random.sample(replay_buffer, 64)
            boards, rows, cols, targets = zip(*batch)
            graph_batch = Batch.from_data_list([create_graph(b).to(device) for b in boards])
            
            model.train()
            optimizer.zero_grad()
            q_preds = model(graph_batch).squeeze()
            indices = torch.tensor([i * 64 + r * 8 + c for i, (r, c) in enumerate(zip(rows, cols))], dtype=torch.long).to(device)
            q_preds_selected = q_preds[indices]
            q_targets = torch.tensor(targets, dtype=torch.float).to(device)
            
            loss = F.mse_loss(q_preds_selected, q_targets)
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"Monte Carlo Episode {episode} completed")

    torch.save(model.state_dict(), 'gnn_montecarlo.pth')
    print("Saved GNN + Monte Carlo model")

# Monte Carlo Simulation (unchanged)
def monte_carlo_simulation(game, move, simulations=10):
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

# Test GNN vs Minimax
def test_vs_minimax(gnn_model, num_games=10):
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

if __name__ == "__main__":
    # Train GNN + Q-Learning
    # gnn_qlearning = GNNQLearning().to(device)
    # optimizer_qlearning = torch.optim.Adam(gnn_qlearning.parameters(), lr=0.001)
    # train_gnn_qlearning(gnn_qlearning, optimizer_qlearning)
    # qlearning_win_rate = test_vs_minimax(gnn_qlearning)
    # print(f"GNN + Q-Learning win rate vs Minimax: {qlearning_win_rate:.2%}")

    # Train GNN + Monte Carlo
    gnn_montecarlo = GNNMonteCarlo().to(device)
    optimizer_montecarlo = torch.optim.Adam(gnn_montecarlo.parameters(), lr=0.001)
    train_gnn_montecarlo(gnn_montecarlo, optimizer_montecarlo)
    montecarlo_win_rate = test_vs_minimax(gnn_montecarlo)
    print(f"GNN + Monte Carlo win rate vs Minimax: {montecarlo_win_rate:.2%}")