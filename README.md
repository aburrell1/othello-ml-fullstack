# Othello with RL Agents

Welcome to **Othello with RL Agents**, a web-based implementation of the classic Othello (Reversi) game featuring advanced AI opponents powered by Graph Neural Networks (GNNs) and Minimax algorithms. Play against three AI players—GNN + Q-Learning, GNN + Monte Carlo, and Minimax—or enjoy a two-player mode. This project showcases machine learning and game AI in a sleek, interactive interface.

## Features

- **Game Modes**:
  - **1-Player**: Compete against one of three AI opponents.
  - **2-Player**: Play locally against another human with an undo feature.
- **AI Opponents**:
  - **GNN + Q-Learning**: A robust GNN trained with Q-Learning for strategic play.
  - **GNN + Monte Carlo**: Combines GNN predictions with Monte Carlo simulations.
  - **Minimax**: A traditional depth-5 Minimax algorithm with heuristics.
- **Interactive UI**:
  - Larger 8x8 board (640x640px) with red valid move highlights.
  - Game-over screen displaying the winner and final score.
  - Styled title with green and black colors.
- **Backend**: Flask-powered API for game logic and AI moves.
- **Frontend**: React-based interface.