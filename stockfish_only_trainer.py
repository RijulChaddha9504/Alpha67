#!/usr/bin/env python3
"""
Stockfish-only continuous trainer with performance test vs previous snapshot.

Phases per cycle:
  1) Play N games: current model vs Stockfish -> collect training positions
  2) Train the model for some batches
  3) Evaluate: current (post-train) model vs previous snapshot (pre-train) for M games
Runs for a user-specified number of minutes.
"""

import argparse
import time
import random
import json
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine

from board_state import board_to_tensor
from network_architecture import ChessNetwork
from MCTSAlgorithm import MCTS, MCTSNode
from config import MODEL_PATH

# -------------------------
# Helper: safe Stockfish init
# -------------------------
def init_stockfish(path, skill_level=1):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            engine.configure({"Skill Level": int(skill_level)})
        except Exception:
            pass
        try:
            engine.configure({"UCI_LimitStrength": True})
        except Exception:
            pass
        try:
            requested_elo = 800 + (int(skill_level) * 100)
            requested_elo = max(requested_elo, 1320)
            engine.configure({"UCI_Elo": int(requested_elo)})
        except Exception:
            pass
        return engine
    except Exception as e:
        print(f"⚠ Could not start Stockfish: {e}")
        return None

# -------------------------
# Trainer class
# -------------------------
class StockfishTrainer:
    def __init__(self,
                 stockfish_path="dist/stockfish",
                 buffer_size=50000,
                 batch_size=128,
                 num_simulations=100,
                 learning_rate=3e-4,
                 device=None,
                 checkpoint_dir="checkpoints",
                 torch_threads=None):
        # Device
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"🔧 Device: {self.device}")

        # Optionally set threads (helpful on CPU-only systems)
        if torch_threads:
            torch.set_num_threads(int(torch_threads))
            print(f"🔧 Torch threads set to {torch_threads}")

        # Checkpoint dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Model + optimizer + scheduler
        self.network = ChessNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # Replay buffer and parameters
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_simulations = num_simulations

        # Stockfish
        self.stockfish_path = stockfish_path
        self.stockfish_engine = None

        # Stats
        self.iteration = 0
        self.stats = {
            "start_time": time.time(),
            "total_games": 0,
            "training_steps": 0,
            "recent_losses": []
        }

        # Try load latest model if available
        self._load_latest_model_if_exists()

    # -----------------------------------------
    # Play: model vs Stockfish (collect training data)
    # -----------------------------------------
    def play_game_vs_stockfish(self, bot_is_white=True, stockfish_time_limit=0.1):
        board = chess.Board()
        game_positions = []
        max_moves = 200
        move_count = 0

        # We'll use a local MCTS instance for this game (fresh statistics)
        mcts = MCTS(self.network, num_simulations=self.num_simulations, device=self.device)

        while not board.is_game_over() and move_count < max_moves:
            is_bot_turn = (board.turn == chess.WHITE) == bot_is_white
            if is_bot_turn:
                root = MCTSNode(board)
                if not root.board.is_game_over():
                    mcts._expand_node(root)

                # run simulations
                for _ in range(self.num_simulations):
                    node = root
                    path = [node]
                    while not node.is_leaf() and not node.board.is_game_over():
                        node = mcts._select_child(node)
                        path.append(node)

                    if not node.board.is_game_over():
                        if node.is_leaf():
                            value = mcts._expand_node(node)
                        else:
                            t = board_to_tensor(node.board)
                            t = torch.from_numpy(t).unsqueeze(0).float().to(self.device)
                            with torch.no_grad():
                                _, value_out = self.network(t)
                            value = value_out[0].item()
                    else:
                        result = node.board.result()
                        value = mcts._result_to_value(result, node.board.turn)

                    mcts._backpropagate(path, value)

                # collect move probabilities from root visit counts
                move_probs = np.zeros(4096, dtype=np.float32)
                total_visits = sum(child.visit_count for child in root.children.values())
                if total_visits > 0:
                    for move, child in root.children.items():
                        idx = move.from_square * 64 + move.to_square
                        move_probs[idx] = child.visit_count / total_visits

                board_state = board_to_tensor(board)
                game_positions.append({"board": board_state, "move_probs": move_probs, "turn": board.turn})

                selected_move = mcts._best_move(root)
            else:
                # Stockfish move
                try:
                    result = self.stockfish_engine.play(board, chess.engine.Limit(time=stockfish_time_limit))
                    selected_move = result.move
                except Exception as e:
                    # fallback
                    selected_move = random.choice(list(board.legal_moves))
                    print(f"Stockfish play error, random fallback: {e}")

            board.push(selected_move)
            move_count += 1

        # annotate outcome
        result = board.result()
        for pos in game_positions:
            if result == "1-0":
                pos["outcome"] = 1.0 if pos["turn"] else -1.0
            elif result == "0-1":
                pos["outcome"] = -1.0 if pos["turn"] else 1.0
            else:
                pos["outcome"] = 0.0

        return game_positions, result, move_count

    # -----------------------------------------
    # Small helper: save current model as both main MODEL_PATH and an iteration snapshot
    # -----------------------------------------
    def _save_model_snapshot(self, snapshot_name=None):
        if snapshot_name is None:
            snapshot_name = f"snapshot_iter_{self.iteration}.pth"
        snapshot_path = self.checkpoint_dir / snapshot_name
        torch.save(self.network.state_dict(), snapshot_path)
        # also update canonical MODEL_PATH
        torch.save(self.network.state_dict(), MODEL_PATH)
        return snapshot_path

    # -----------------------------------------
    # Load latest model if exists (for resume)
    # -----------------------------------------
    def _load_latest_model_if_exists(self):
        main_path = Path(MODEL_PATH)
        if main_path.exists():
            try:
                sd = torch.load(main_path, map_location=self.device)
                self.network.load_state_dict(sd)
                print(f"✓ Loaded main model from {main_path}")
            except Exception as e:
                print(f"⚠ Could not load model: {e}")

    # -----------------------------------------
    # Training on batch (same as yours but small tweaks)
    # -----------------------------------------
    def train_on_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)

        boards = torch.tensor([d['board'] for d in batch], dtype=torch.float32).to(self.device)
        target_probs = torch.tensor([d['move_probs'] for d in batch], dtype=torch.float32).to(self.device)
        target_values = torch.tensor([[d['outcome']] for d in batch], dtype=torch.float32).to(self.device)

        self.network.train()
        pred_probs, pred_values = self.network(boards)

        policy_loss = -torch.mean(torch.sum(target_probs * torch.log_softmax(pred_probs, dim=1), dim=1))
        value_loss = nn.MSELoss()(pred_values, target_values)
        total_loss = 0.7 * policy_loss + 0.3 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.stats['training_steps'] += 1
        return {'total_loss': total_loss.item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}

    # -----------------------------------------
    # Evaluation: play current model vs a snapshot (frozen) for N games
    # -----------------------------------------
    def evaluate_vs_snapshot(self, snapshot_path, num_games=10, simulations_current=None, simulations_snapshot=None, time_limit=0.05):
        if not Path(snapshot_path).exists():
            print(f"⚠ Snapshot {snapshot_path} not found; skipping evaluation")
            return None

        # Load snapshot into a separate network (frozen)
        snapshot_net = ChessNetwork().to(self.device)
        snapshot_net.load_state_dict(torch.load(snapshot_path, map_location=self.device))
        snapshot_net.eval()

        # Create MCTS instances (one for current, one for snapshot)
        sim_curr = simulations_current or self.num_simulations
        sim_prev = simulations_snapshot or max(10, int(self.num_simulations * 0.5))

        wins = 0.0
        draws = 0
        losses = 0.0
        moves_list = []

        for i in range(num_games):
            board = chess.Board()
            move_count = 0
            max_moves = 200

            mcts_curr = MCTS(self.network, num_simulations=sim_curr, device=self.device)
            mcts_prev = MCTS(snapshot_net, num_simulations=sim_prev, device=self.device)

            # Alternate colors: half games curr white, half black
            curr_is_white = (i % 2 == 0)

            while not board.is_game_over() and move_count < max_moves:
                if (board.turn == chess.WHITE) == curr_is_white:
                    root = MCTSNode(board)
                    if not root.board.is_game_over():
                        mcts_curr._expand_node(root)

                    for _ in range(sim_curr):
                        node = root
                        path = [node]
                        while not node.is_leaf() and not node.board.is_game_over():
                            node = mcts_curr._select_child(node)
                            path.append(node)
                        if not node.board.is_game_over():
                            if node.is_leaf():
                                val = mcts_curr._expand_node(node)
                            else:
                                t = board_to_tensor(node.board)
                                t = torch.from_numpy(t).unsqueeze(0).float().to(self.device)
                                with torch.no_grad():
                                    _, vout = self.network(t)
                                val = vout[0].item()
                        else:
                            val = mcts_curr._result_to_value(node.board.result(), node.board.turn)
                        mcts_curr._backpropagate(path, val)

                    move = mcts_curr._best_move(root)
                else:
                    root = MCTSNode(board)
                    if not root.board.is_game_over():
                        mcts_prev._expand_node(root)

                    for _ in range(sim_prev):
                        node = root
                        path = [node]
                        while not node.is_leaf() and not node.board.is_game_over():
                            node = mcts_prev._select_child(node)
                            path.append(node)
                        if not node.board.is_game_over():
                            if node.is_leaf():
                                val = mcts_prev._expand_node(node)
                            else:
                                t = board_to_tensor(node.board)
                                t = torch.from_numpy(t).unsqueeze(0).float().to(self.device)
                                with torch.no_grad():
                                    _, vout = snapshot_net(t)
                                val = vout[0].item()
                        else:
                            val = mcts_prev._result_to_value(node.board.result(), node.board.turn)
                        mcts_prev._backpropagate(path, val)

                    move = mcts_prev._best_move(root)

                board.push(move)
                move_count += 1

            res = board.result()
            moves_list.append(move_count)
            if res == "1-0":
                if curr_is_white:
                    wins += 1
                else:
                    losses += 1
            elif res == "0-1":
                if curr_is_white:
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1

        total = num_games
        winrate = (wins + 0.5 * draws) / total
        stats = {
            "games": total,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "winrate": winrate,
            "avg_moves": float(np.mean(moves_list))
        }
        return stats

    # -----------------------------------------
    # Main loop: run for minutes_limit minutes
    # -----------------------------------------
    def run_for_minutes(self,
                        minutes_limit=30,
                        phase1_games=4,
                        phase2_games=10,
                        training_batches=40,
                        stockfish_skill=1,
                        stockfish_time_limit=0.08,
                        save_every=1):
        end_time = time.time() + (minutes_limit * 60)
        cycle = 0

        # Start stockfish once and reuse (faster)
        self.stockfish_engine = init_stockfish(self.stockfish_path, skill_level=stockfish_skill)
        if not self.stockfish_engine:
            print("⚠ Stockfish not available — exiting.")
            return

        try:
            while time.time() < end_time:
                cycle += 1
                self.iteration += 1
                print("\n" + "=" * 60)
                print(f"🔁 Cycle {cycle} (iteration {self.iteration}) start: {datetime.now().isoformat()}")
                print("=" * 60)

                # 1) Save a snapshot of the current model BEFORE training (for later evaluation)
                prev_snapshot = self._save_model_snapshot(snapshot_name=f"snapshot_pre_iter_{self.iteration}.pth")
                print(f"💾 Saved pre-train snapshot: {prev_snapshot}")

                # 2) Phase 1: Play model vs Stockfish to collect data
                print(f"\n⚔️ Phase 1: Playing {phase1_games} games vs Stockfish (skill {stockfish_skill})")
                wins = 0.0
                for g in range(phase1_games):
                    bot_white = (g % 2 == 0)
                    positions, result, moves = self.play_game_vs_stockfish(bot_is_white=bot_white, stockfish_time_limit=stockfish_time_limit)
                    self.replay_buffer.extend(positions)
                    self.stats['total_games'] += 1
                    # track win vs stockfish
                    if (result == "1-0" and bot_white) or (result == "0-1" and not bot_white):
                        wins += 1
                    elif result == "1/2-1/2":
                        wins += 0.5
                    print(f"  Game {g+1}/{phase1_games} ({'White' if bot_white else 'Black'}): {result} in {moves} moves - {len(positions)} positions")
                winrate_sf = wins / phase1_games if phase1_games > 0 else 0
                print(f"  📊 Phase1 winrate vs Stockfish: {winrate_sf*100:.1f}%")

                # 3) Train for training_batches iterations
                print(f"\n📚 Training: {training_batches} batches (buffer size {len(self.replay_buffer)})")
                losses = []
                for b in range(training_batches):
                    loss = self.train_on_batch()
                    if loss:
                        losses.append(loss['total_loss'])
                    if (b + 1) % max(1, training_batches // 4) == 0:
                        recent = f"{losses[-1]:.4f}" if losses else "N/A"
                        print(f"  Training batch {b+1}/{training_batches} - recent loss: {recent}")

                if losses:
                    avg_loss = float(np.mean(losses))
                    self.stats['recent_losses'].append(avg_loss)
                    if len(self.stats['recent_losses']) > 10:
                        self.stats['recent_losses'].pop(0)
                    self.scheduler.step(avg_loss)
                    print(f"  📈 Avg training loss: {avg_loss:.4f}")

                # After training, save an updated model
                updated_snapshot = self._save_model_snapshot(snapshot_name=f"snapshot_post_iter_{self.iteration}.pth")
                print(f"💾 Saved post-train model: {updated_snapshot}")

                # 4) Phase 2: Evaluate current (post-train) vs pre-train snapshot (phase measurement)
                print(f"\n🏁 Phase 2: Evaluate current vs pre-train snapshot for {phase2_games} games")
                eval_stats = self.evaluate_vs_snapshot(prev_snapshot, num_games=phase2_games)
                if eval_stats:
                    print(f"  Eval result: {eval_stats['wins']}W / {eval_stats['draws']}D / {eval_stats['losses']}L | winrate={(eval_stats['winrate']*100):.1f}% | avg moves {eval_stats['avg_moves']:.1f}")
                else:
                    print("  No eval stats (snapshot missing)")

                # periodically save a full checkpoint with optimizer+buffer
                if (self.iteration % save_every) == 0:
                    checkpoint = {
                        'iteration': self.iteration,
                        'network_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'stats': self.stats,
                        'replay_buffer': list(self.replay_buffer)
                    }
                    ck_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
                    torch.save(checkpoint, ck_path)
                    print(f"💾 Full checkpoint saved: {ck_path}")

                # log quick summary
                print(f"\nCycle {cycle} complete. Total games so far: {self.stats['total_games']}. Training steps: {self.stats['training_steps']}")

        except KeyboardInterrupt:
            print("⚠ Interrupted by user, saving model and exiting...")

        finally:
            # cleanup
            if self.stockfish_engine:
                self.stockfish_engine.quit()
            print("Done. Final model saved to model path.")
            torch.save(self.network.state_dict(), MODEL_PATH)
            print(f"Model at {MODEL_PATH}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30, help="Run time in minutes")
    parser.add_argument("--phase1-games", type=int, default=4, help="Model vs Stockfish games per cycle")
    parser.add_argument("--phase2-games", type=int, default=10, help="Model vs previous snapshot games per cycle (evaluation)")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--training-batches", type=int, default=40)
    parser.add_argument("--stockfish-path", type=str, default="dist/stockfish")
    parser.add_argument("--stockfish-skill", type=int, default=1)
    parser.add_argument("--stockfish-time", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    trainer = StockfishTrainer(
        stockfish_path=args.stockfish_path,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_simulations=args.simulations,
        device=None,
        checkpoint_dir="checkpoints",
        torch_threads=args.torch_threads
    )

    trainer.run_for_minutes(
        minutes_limit=args.minutes,
        phase1_games=args.phase1_games,
        phase2_games=args.phase2_games,
        training_batches=args.training_batches,
        stockfish_skill=args.stockfish_skill,
        stockfish_time_limit=args.stockfish_time
    )

if __name__ == "__main__":
    main()
