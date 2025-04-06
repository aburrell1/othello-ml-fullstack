import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function Game() {
  const [screen, setScreen] = useState('title'); 
  const [board, setBoard] = useState([]);
  const [currentPlayer, setCurrentPlayer] = useState(2);
  const [gameOver, setGameOver] = useState(false);
  const [message, setMessage] = useState('');
  const [gameMode, setGameMode] = useState('1player'); // null, '1player', '2player'
  const [playerColor, setPlayerColor] = useState('black'); // null, 'black', 'white'
  const [aiType, setAiType] = useState('minimax'); // null, 'gnn_qlearning', 'gnn_montecarlo', 'minimax'
  const [blackCount, setBlackCount] = useState(2);
  const [whiteCount, setWhiteCount] = useState(2);
  const [validMoves, setValidMoves] = useState([]);
  const [hoveredCell, setHoveredCell] = useState(null);

  useEffect(() => {
    if (screen === 'play') fetchBoard();
  }, [screen]);

  const fetchBoard = async () => {
    const response = await axios.get('http://localhost:5000/api/board');
    setBoard(response.data.board);
    setCurrentPlayer(response.data.currentPlayer);
    setGameOver(response.data.gameOver);
    setBlackCount(response.data.blackCount);
    setWhiteCount(response.data.whiteCount);
    setValidMoves(response.data.validMoves);
    updateMessage(response.data);
  };

  const updateMessage = (data) => {
    if (data.gameOver) {
      const winner = blackCount > whiteCount ? 'Black' : whiteCount > blackCount ? 'White' : 'Tie';
      setMessage(`Game Over! Winner: ${winner} (Black: ${blackCount}, White: ${whiteCount})`);
    } else {
      setMessage(`${currentPlayer === 2 ? 'Black' : 'White'} to play`);
    }
  };

  const handlePlayerMove = async (row, col) => {
    if (!gameMode || (gameMode === '1player' && (!aiType || !playerColor)) || gameOver) return;
    if (gameMode === '1player' && currentPlayer !== (playerColor === 'black' ? 2 : 1)) return;

    const response = await axios.post('http://localhost:5000/api/move', { row, col });
    if (response.data.success) {
      setBoard(response.data.board);
      setCurrentPlayer(response.data.currentPlayer);
      setGameOver(response.data.gameOver);
      setBlackCount(response.data.blackCount);
      setWhiteCount(response.data.whiteCount);
      setValidMoves(response.data.validMoves);
      if (gameMode === '1player' && !response.data.gameOver) {
        setMessage('Your move! AI is thinking...');
        await handleAIMove();
      } else {
        updateMessage(response.data);
      }
    } else {
      setMessage('Invalid move!');
    }
  };

  const handleAIMove = async () => {
    const response = await axios.post('http://localhost:5000/api/ai-move');
    setBoard(response.data.board);
    setCurrentPlayer(response.data.currentPlayer);
    setGameOver(response.data.gameOver);
    setBlackCount(response.data.blackCount);
    setWhiteCount(response.data.whiteCount);
    setValidMoves(response.data.validMoves);
    if (response.data.success) {
      setMessage(`AI moved to (${response.data.row}, ${response.data.col})`);
    } else if (response.data.skipped) {
      setMessage(response.data.message);
    }
    updateMessage(response.data);
  };

  const handleReset = async () => {
    await axios.post('http://localhost:5000/api/reset');
    setScreen('title');
    setGameMode(null);
    setPlayerColor(null);
    setAiType(null);
  };

  const handleRestart = async () => {
    await axios.post('http://localhost:5000/api/reset');
    setScreen('title');
  };

  const handleUndo = async () => {
    const response = await axios.post('http://localhost:5000/api/undo');
    if (response.data.success) {
      setBoard(response.data.board);
      setCurrentPlayer(response.data.currentPlayer);
      setGameOver(response.data.gameOver);
      setBlackCount(response.data.blackCount);
      setWhiteCount(response.data.whiteCount);
      setValidMoves(response.data.validMoves);
      updateMessage(response.data);
    } else {
      setMessage('No moves to undo!');
    }
  };

  const handleSetAI = async (type) => {
    setAiType(type);
    await axios.post('http://localhost:5000/api/set-ai', { aiType: type });
  };

  const renderCell = (value) => {
    if (value === 0) return '';
    if (value === 1) return '⚪';
    if (value === 2) return '⚫';
  };

  const isValidMove = (row, col) => {
    if (gameMode === '1player' && currentPlayer !== (playerColor === 'black' ? 2 : 1)) return false;
    return validMoves.some(([r, c]) => r === row && c === col);
  };

  return (
    <div className="game-container">
      {screen === 'title' && (
        <div className="title-screen">
          <h1 className="title">
            <span className="green-text">Othello:</span> <span className="black-text">GNN vs Minimax</span>
          </h1>
          <button className="title-button" onClick={() => setScreen('play')}>
            Play
          </button>
          <button className="title-button" onClick={() => setScreen('settings')}>
            Settings
          </button>
        </div>
      )}
      {screen === 'play' && (
        <div className="play-screen">
          {!gameOver ? (
            <div className="board">
              {board.map((row, i) => (
                <div key={i} className="row">
                  {row.map((cell, j) => (
                    <div
                      key={j}
                      className={`cell ${isValidMove(i, j) ? 'valid-move' : ''}`}
                      onClick={() => handlePlayerMove(i, j)}
                      onMouseEnter={() => setHoveredCell([7 - i, j])}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      {renderCell(cell)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <div className="game-over-square">
              <h2>{message}</h2>
            </div>
          )}
          <div className="counters">
            <span>Black: {blackCount}</span>
            <span>White: {whiteCount}</span>
          </div>
          <p>{message}</p>
          {hoveredCell && !gameOver && (
            <p className="coordinates">Coordinates: ({hoveredCell[0]}, {hoveredCell[1]})</p>
          )}
          {gameMode === '2player' && !gameOver && (
            <button className="back-button" onClick={handleUndo}>
              Undo
            </button>
          )}
          <button className="back-button" onClick={handleRestart}>
            Restart
          </button>
        </div>
      )}
      {screen === 'settings' && (
        <div className="settings-screen">
          <h2>Settings</h2>
          <div className="setting-group">
            <h3>Game Mode</h3>
            <button
              className={`mode-button ${gameMode === '1player' ? 'selected' : ''}`}
              onClick={() => setGameMode('1player')}
            >
              1 Player
            </button>
            <button
              className={`mode-button ${gameMode === '2player' ? 'selected' : ''}`}
              onClick={() => setGameMode('2player')}
            >
              2 Player
            </button>
          </div>
          {gameMode === '1player' && (
            <>
              <div className="setting-group">
                <h3>AI Opponent</h3>
                <button
                  className={`ai-button ${aiType === 'gnn_qlearning' ? 'selected' : ''}`}
                  onClick={() => handleSetAI('gnn_qlearning')}
                >
                  GNN + Q-Learning
                </button>
                <button
                  className={`ai-button ${aiType === 'gnn_montecarlo' ? 'selected' : ''}`}
                  onClick={() => handleSetAI('gnn_montecarlo')}
                >
                  GNN + Monte Carlo
                </button>
                <button
                  className={`ai-button ${aiType === 'minimax' ? 'selected' : ''}`}
                  onClick={() => handleSetAI('minimax')}
                >
                  Minimax Player
                </button>
              </div>
              <div className="setting-group">
                <h3>Player Color</h3>
                <button
                  className={`mode-button ${playerColor === 'black' ? 'selected' : ''}`}
                  onClick={() => setPlayerColor('black')}
                >
                  Play as Black
                </button>
                <button
                  className={`mode-button ${playerColor === 'white' ? 'selected' : ''}`}
                  onClick={() => setPlayerColor('white')}
                >
                  Play as White
                </button>
              </div>
            </>
          )}
          <button className="back-button" onClick={() => setScreen('title')}>
            Back
          </button>
        </div>
      )}
    </div>
  );
}

export default Game;