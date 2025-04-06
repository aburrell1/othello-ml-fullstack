import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from app import Othello, create_graph, GNNQLearning, GNNMonteCarlo  # Import from app.py

# Training function
def train_gnn_qlearning(model, optimizer, episodes=1000):
    game = Othello()
    for episode in range(episodes):
        game.reset()
        while not game.is_game_over():
            graph_data = create_graph(game.board)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
            model.eval()
            with torch.no_grad():
                q_values = model(graph_data).squeeze()
            valid_q_values = [q_values[row * 8 + col].item() for row, col in valid_moves]
            best_move_idx = np.argmax(valid_q_values)
            row, col = valid_moves[best_move_idx]
            
            game.make_move(row, col)
            next_game = game.copy()
            next_valid_moves = next_game.get_valid_moves()
            reward = 1 if next_valid_moves else -1
            if next_game.is_game_over():
                black_count, white_count = next_game.get_counts()
                reward = -(10 * (black_count - white_count)) if black_count > white_count else (10 * (white_count - black_count))
            
            model.train()
            optimizer.zero_grad()
            q_pred = model(graph_data)[row * 8 + col]
            next_q_values = model(create_graph(next_game.board)).squeeze()
            q_target = reward + 0.9 * (max([next_q_values[m[0] * 8 + m[1]] for m in next_valid_moves], default=0) if next_valid_moves else 0)
            loss = F.mse_loss(q_pred, torch.tensor(q_target, dtype=torch.float))
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"Q-Learning Episode {episode} completed")
    
    torch.save(model.state_dict(), 'gnn_qlearning.pth')
    print("Saved GNN + Q-Learning model")

def train_gnn_montecarlo(model, optimizer, episodes=1000, simulations=50):
    game = Othello()
    for episode in range(episodes):
        game.reset()
        while not game.is_game_over():
            graph_data = create_graph(game.board)
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            
            model.eval()
            with torch.no_grad():
                q_values = model(graph_data).squeeze()
            valid_q_values = [(q_values[row * 8 + col].item() + monte_carlo_simulation(game, [row, col], simulations)) for row, col in valid_moves]
            best_move_idx = np.argmax(valid_q_values)
            row, col = valid_moves[best_move_idx]
            
            game.make_move(row, col)
            reward = 1 if game.get_valid_moves() else -1
            if game.is_game_over():
                black_count, white_count = game.get_counts()
                reward = -(10 * (black_count - white_count)) if black_count > white_count else (10 * (white_count - black_count))
            
            model.train()
            optimizer.zero_grad()
            q_pred = model(graph_data)[row * 8 + col]
            loss = F.mse_loss(q_pred, torch.tensor(reward, dtype=torch.float))
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"Monte Carlo Episode {episode} completed")
    
    torch.save(model.state_dict(), 'gnn_montecarlo.pth')
    print("Saved GNN + Monte Carlo model")

def monte_carlo_simulation(game, move, simulations):
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

if __name__ == "__main__":
    # Train GNN + Q-Learning
    gnn_qlearning = GNNQLearning()
    optimizer_qlearning = torch.optim.Adam(gnn_qlearning.parameters(), lr=0.001)
    train_gnn_qlearning(gnn_qlearning, optimizer_qlearning, episodes=1000)

    # Train GNN + Monte Carlo
    gnn_montecarlo = GNNMonteCarlo()
    optimizer_montecarlo = torch.optim.Adam(gnn_montecarlo.parameters(), lr=0.001)
    train_gnn_montecarlo(gnn_montecarlo, optimizer_montecarlo, episodes=1000, simulations=50)