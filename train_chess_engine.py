#!/usr/bin/env python3
# train_chess_engine.py - AlphaZero-style self-play training

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import random
from collections import deque
from chess_engine import ChessEngine
from board_state import board_to_tensor
from network_architecture import ChessNetwork
from MCTSAlgorithm import MCTS, MCTSNode
from config import MODEL_PATH
import os


class AlphaZeroTrainer:
    """
    Self-play training loop for chess engine.
    
    Training cycle:
    1. Self-play: Generate games using current network + MCTS
    2. Training: Update network on self-play data
    3. Evaluation: Test new network vs old network
    4. Iteration: Repeat
    """
    
    def __init__(self, 
                 network=None,
                 buffer_size=10000,
                 batch_size=256,
                 num_simulations=100,
                 learning_rate=0.001,
                 device=None):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize or load network
        self.network = network if network else ChessNetwork()
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # MCTS for self-play
        self.mcts = MCTS(self.network, num_simulations=num_simulations, device=self.device)
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        
    def generate_self_play_game(self, temperature=1.0):
        """
        Play one full game using MCTS, collecting training data.
        
        Returns:
            List of (board_state, move_probs, outcome) tuples
        """
        board = chess.Board()
        game_data = []
        
        move_count = 0
        
        while not board.is_game_over() and move_count < 200:  # Max 200 moves
            # Run MCTS to get move probabilities
            root = MCTSNode(board)
            
            # Expand root
            if not root.board.is_game_over():
                self.mcts._expand_node(root)
            
            # Run simulations
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]
                
                # SELECT
                while not node.is_leaf() and not node.board.is_game_over():
                    node = self.mcts._select_child(node)
                    search_path.append(node)
                
                # EXPAND & EVALUATE
                if not node.board.is_game_over():
                    if node.is_leaf():
                        value = self.mcts._expand_node(node)
                    else:
                        board_tensor = board_to_tensor(node.board)
                        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).float().to(self.device)
                        with torch.no_grad():
                            _, value_output = self.network(board_tensor)
                        value = value_output[0].item()
                else:
                    result = node.board.result()
                    value = self.mcts._result_to_value(result, node.board.turn)
                
                # BACKPROPAGATE
                self.mcts._backpropagate(search_path, value)
            
            # Create move probability distribution from visit counts
            move_probs = np.zeros(4096, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())
            
            if total_visits > 0:
                for move, child in root.children.items():
                    move_idx = move.from_square * 64 + move.to_square
                    # Apply temperature
                    if temperature == 0:
                        # Deterministic (argmax)
                        move_probs[move_idx] = 1.0 if child.visit_count == max(c.visit_count for c in root.children.values()) else 0.0
                    else:
                        # Stochastic (proportional to visits^(1/T))
                        move_probs[move_idx] = (child.visit_count / total_visits) ** (1 / temperature)
                
                # Renormalize
                if move_probs.sum() > 0:
                    move_probs /= move_probs.sum()
            
            # Store training data (board state, move probs, outcome will be filled later)
            board_state = board_to_tensor(board)
            game_data.append({
                'board': board_state,
                'move_probs': move_probs,
                'turn': board.turn
            })
            
            # Select move (stochastic early game, deterministic late game)
            if move_count < 30:
                # Sample from distribution
                legal_move_probs = {move: child.visit_count / total_visits 
                                   for move, child in root.children.items()}
                moves = list(legal_move_probs.keys())
                probs = list(legal_move_probs.values())
                selected_move = random.choices(moves, weights=probs)[0]
            else:
                # Pick best move
                selected_move = self.mcts._best_move(root)
            
            board.push(selected_move)
            move_count += 1
        
        # Game over - assign outcome to all positions
        result = board.result()
        
        for data in game_data:
            if result == "1-0":
                outcome = 1.0 if data['turn'] else -1.0
            elif result == "0-1":
                outcome = -1.0 if data['turn'] else 1.0
            else:
                outcome = 0.0
            
            data['outcome'] = outcome
        
        return game_data, result
    
    def train_on_batch(self):
        """
        Sample a batch from replay buffer and train the network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare tensors
        boards = torch.tensor([d['board'] for d in batch], dtype=torch.float32).to(self.device)
        target_probs = torch.tensor([d['move_probs'] for d in batch], dtype=torch.float32).to(self.device)
        target_values = torch.tensor([[d['outcome']] for d in batch], dtype=torch.float32).to(self.device)
        
        # Forward pass
        self.network.train()
        pred_probs, pred_values = self.network(boards)
        
        # Loss calculation
        # Policy loss: cross-entropy between MCTS move distribution and network output
        policy_loss = -torch.mean(torch.sum(target_probs * torch.log_softmax(pred_probs, dim=1), dim=1))
        
        # Value loss: MSE between game outcome and network prediction
        value_loss = nn.MSELoss()(pred_values, target_values)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def run_training_iteration(self, num_games=50, num_train_steps=100):
        """
        Run one complete training iteration:
        1. Generate self-play games
        2. Train on the data
        """
        print(f"\n{'='*60}")
        print(f"Starting training iteration: {num_games} games, {num_train_steps} train steps")
        print(f"{'='*60}\n")
        
        # 1. Self-play
        print("Phase 1: Self-play...")
        for game_num in range(num_games):
            game_data, result = self.generate_self_play_game(temperature=1.0)
            
            # Add to replay buffer
            self.replay_buffer.extend(game_data)
            
            print(f"Game {game_num + 1}/{num_games}: {len(game_data)} positions, Result: {result}")
        
        print(f"\nReplay buffer size: {len(self.replay_buffer)}")
        
        # 2. Training
        print("\nPhase 2: Training...")
        total_losses = []
        
        for step in range(num_train_steps):
            loss_dict = self.train_on_batch()
            
            if loss_dict:
                total_losses.append(loss_dict['total_loss'])
                
                if (step + 1) % 20 == 0:
                    print(f"Step {step + 1}/{num_train_steps}: "
                          f"Loss={loss_dict['total_loss']:.4f} "
                          f"(Policy={loss_dict['policy_loss']:.4f}, "
                          f"Value={loss_dict['value_loss']:.4f})")
        
        if total_losses:
            print(f"\nAverage loss: {np.mean(total_losses):.4f}")
        
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path=MODEL_PATH):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_size': len(self.replay_buffer)
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path=MODEL_PATH):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded from {path}")
            return True
        return False


def main():
    """
    Main training loop with command-line options
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Chess Engine using AlphaZero')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=10, help='Games per iteration')
    parser.add_argument('--train-steps', type=int, default=50, help='Training steps per iteration')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N iterations')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (5 iterations, 3 games)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        print("⚡ QUICK TEST MODE")
        args.iterations = 5
        args.games = 3
        args.train_steps = 20
        args.simulations = 50
    
    print("="*60)
    print("AlphaZero Chess Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Total iterations: {args.iterations}")
    print(f"  Games per iteration: {args.games}")
    print(f"  Training steps per iteration: {args.train_steps}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Learning rate: {args.lr}")
    print("="*60 + "\n")
    
    trainer = AlphaZeroTrainer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_simulations=args.simulations,
        learning_rate=args.lr
    )
    
    # Load existing checkpoint if resuming
    if args.resume:
        if trainer.load_checkpoint():
            print("✓ Resumed from checkpoint\n")
        else:
            print("⚠ No checkpoint found, starting fresh\n")
    
    # Training loop
    try:
        for iteration in range(args.iterations):
            print(f"\n{'#'*60}")
            print(f"### ITERATION {iteration + 1}/{args.iterations}")
            print(f"{'#'*60}")
            
            # Run one iteration
            trainer.run_training_iteration(
                num_games=args.games,
                num_train_steps=args.train_steps
            )
            
            # Save checkpoint periodically
            if (iteration + 1) % args.save_every == 0:
                trainer.save_checkpoint()
        
        # Final save
        trainer.save_checkpoint()
        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("✓ Checkpoint saved. Resume with --resume flag")


if __name__ == "__main__":
    main()