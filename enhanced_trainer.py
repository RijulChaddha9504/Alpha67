#!/usr/bin/env python3
"""
Speed-optimized Enhanced Stockfish Trainer (full refactor for fast Phase C + GPU-first).

Improvements:
 - Cached initial snapshot net
 - Snapshot side uses fewer simulations by default (configurable)
 - torch.no_grad() and eval() used for all evaluations
 - eval_batch_size passed to MCTS; increase for better GPU utilization
 - Fast-eval mode (silence IO) for Phase C
 - Optional parallel evaluation (use cautiously with GPU)
 - CLI flags for controlling snapshot sims, fast-eval, and parallel eval workers
 - Defaults tuned for GPU use
"""
import argparse
import time
import random
import shutil
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime
import multiprocessing as mp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import chess.engine

from board_state import board_to_tensor
from network_architecture import ChessNetwork
from MCTSAlgorithm import MCTS, MCTSNode
from config import MODEL_PATH

# -------------------------
# Helpers
# -------------------------

def init_stockfish(path, skill_level=1, threads=None):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        cfg = {}
        try:
            cfg["Skill Level"] = int(skill_level)
        except Exception:
            pass
        try:
            cfg["UCI_LimitStrength"] = True
        except Exception:
            pass
        try:
            requested_elo = 800 + (int(skill_level) * 100)
            requested_elo = max(800, requested_elo)
            cfg["UCI_Elo"] = int(requested_elo)
        except Exception:
            pass
        if threads is not None:
            try:
                cfg["Threads"] = int(threads)
            except Exception:
                pass
        if cfg:
            try:
                engine.configure(cfg)
            except Exception:
                pass
        return engine
    except Exception as e:
        print(f"⚠ Could not start Stockfish at '{path}': {e}")
        return None

def classify_game_phase(board: chess.Board) -> str:
    move_count = board.fullmove_number
    piece_count = len(board.piece_map())
    if move_count <= 10:
        return "opening"
    if piece_count <= 10:
        return "endgame"
    return "midgame"

def generate_varied_starting_position(stockfish_engine, phase="random", max_attempts=20):
    for _ in range(max_attempts):
        board = chess.Board()
        ph = phase
        if phase == "random":
            ph = random.choice(["opening", "midgame", "endgame"])
        try:
            if ph == "opening":
                moves = random.randint(3, 8)
                for _ in range(moves):
                    if board.is_game_over():
                        break
                    board.push(random.choice(list(board.legal_moves)))
            elif ph == "midgame":
                moves = random.randint(12, 20)
                for _ in range(moves):
                    if board.is_game_over():
                        break
                    if stockfish_engine:
                        res = stockfish_engine.play(board, chess.engine.Limit(time=0.03))
                        board.push(res.move)
                    else:
                        board.push(random.choice(list(board.legal_moves)))
            elif ph == "endgame":
                for _ in range(40):
                    if board.is_game_over() or len(board.piece_map()) <= 10:
                        break
                    if stockfish_engine:
                        res = stockfish_engine.play(board, chess.engine.Limit(time=0.03))
                        board.push(res.move)
                    else:
                        board.push(random.choice(list(board.legal_moves)))
            if not board.is_game_over():
                return board
        except Exception:
            continue
    return chess.Board()

# -------------------------
# Trainer
# -------------------------
class EnhancedStockfishTrainer:
    def __init__(self,
                 stockfish_path="dist/stockfish-ubuntu-x86-64-avx2",
                 buffer_size=50000,
                 batch_size=128,
                 num_simulations=128,
                 snapshot_sim_fraction=0.25,
                 snapshot_min_sims=8,
                 learning_rate=3e-4,
                 device=None,
                 checkpoint_dir="checkpoints",
                 torch_threads=None,
                 balance_phases=True,
                 show_moves=False,
                 save_pgns=False,
                 pgn_dir="games",
                 self_games_per_cycle=0,
                 mcts_debug=False,
                 stockfish_threads=None,
                 eval_batch_size=None,
                 fast_eval=False,
                 parallel_eval_workers=0):
        # device
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"🔧 Device: {self.device}")

        self.is_cpu = self.device.type == 'cpu'
        if self.is_cpu:
            print("⚠️  CPU detected - performance will be lower")

        if torch_threads:
            torch.set_num_threads(int(torch_threads))
            print(f"🔧 Torch threads set to {torch_threads}")

        # dirs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.pgn_dir = Path(pgn_dir)
        if save_pgns:
            self.pgn_dir.mkdir(parents=True, exist_ok=True)

        # network + optimizer
        self.network = ChessNetwork().to(self.device)
        if hasattr(self.network, "enable_fast_inference"):
            try:
                self.network.enable_fast_inference()
                print("⚡ Fast inference mode enabled on network")
            except Exception:
                pass

        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # replay buffers
        self.balance_phases = balance_phases
        if balance_phases:
            self.replay_buffer = {
                "opening": deque(maxlen=buffer_size // 3),
                "midgame": deque(maxlen=buffer_size // 3),
                "endgame": deque(maxlen=buffer_size // 3)
            }
            print("✅ Phase-balanced training enabled")
        else:
            self.replay_buffer = deque(maxlen=buffer_size)

        # training / mcts params
        self.batch_size = batch_size
        self.num_simulations = int(num_simulations)
        self.snapshot_sim_fraction = float(snapshot_sim_fraction)
        self.snapshot_min_sims = int(snapshot_min_sims)

        self.mcts_eval_batch = eval_batch_size if eval_batch_size else min(128, max(8, int(max(1, self.num_simulations // 4))))
        self.eval_batch_size = self.mcts_eval_batch

        # stockfish
        self.stockfish_path = stockfish_path
        self.stockfish_threads = stockfish_threads
        self.stockfish_engine = None

        # misc flags
        self.show_moves = show_moves
        self.save_pgns = save_pgns
        self.self_games_per_cycle = int(self_games_per_cycle)
        self.mcts_debug = bool(mcts_debug)

        # bookkeeping
        self.iteration = 0
        self.stats = {
            "start_time": time.time(),
            "total_games": 0,
            "training_steps": 0,
            "recent_losses": [],
            "phase1_winrates": [],
            "phase_distribution": defaultdict(int)
        }

        # initial snapshot path & in-memory snapshot net
        self.initial_snapshot_path = self.checkpoint_dir / "snapshot_initial.pth"
        self.initial_snapshot_net = None

        # load main model if present
        self._load_latest_model_if_exists()

        # create initial snapshot if missing (persist and also create in-memory copy)
        try:
            if not self.initial_snapshot_path.exists():
                if Path(MODEL_PATH).exists():
                    shutil.copy2(MODEL_PATH, self.initial_snapshot_path)
                else:
                    torch.save(self.network.state_dict(), self.initial_snapshot_path)
                print(f"🔖 Saved initial snapshot for first-version fights: {self.initial_snapshot_path}")
        except Exception as e:
            print(f"⚠ Failed saving initial snapshot: {e}")

        # attempt to load snapshot into memory
        try:
            if self.initial_snapshot_path.exists():
                snap_net = ChessNetwork().to(self.device)
                sd = torch.load(self.initial_snapshot_path, map_location=self.device)
                snap_net.load_state_dict(sd)
                snap_net.eval()
                for p in snap_net.parameters():
                    p.requires_grad = False
                self.initial_snapshot_net = snap_net
                print("✓ Loaded initial snapshot into memory (cached).")
        except Exception as e:
            print(f"⚠ Could not load initial snapshot network into memory: {e}")
            self.initial_snapshot_net = None

        # fast eval / parallel options
        self.fast_eval = bool(fast_eval)
        self.parallel_eval_workers = int(parallel_eval_workers) if parallel_eval_workers else 0
        if self.parallel_eval_workers > 0:
            print(f"⚠ Parallel eval workers requested: {self.parallel_eval_workers}. Use with caution on GPU.")

    # -------------------------
    # IO / small helpers
    # -------------------------
    def _save_pgn_from_board(self, board: chess.Board, result: str, filename: Path):
        game = chess.pgn.Game()
        node = game
        for mv in board.move_stack:
            node = node.add_variation(mv)
        game.headers["Result"] = result
        with open(filename, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

    def moves_one_line(self, board: chess.Board) -> str:
        temp = chess.Board()
        white_moves = []
        black_moves = []
        for mv in board.move_stack:
            try:
                san = temp.san(mv)
            except Exception:
                san = mv.uci()
            if temp.turn == chess.WHITE:
                white_moves.append(san)
            else:
                black_moves.append(san)
            temp.push(mv)
        white_line = "White: " + (" ".join(white_moves) if white_moves else "(no moves)")
        black_line = "Black: " + (" ".join(black_moves) if black_moves else "(no moves)")
        return white_line + "\n" + black_line

    def _print_moves_from_board(self, board: chess.Board):
        print(self.moves_one_line(board))

    # -------------------------
    # Play a single game vs Stockfish
    # (unchanged but wrapped with eval/no_grad where network is used)
    # -------------------------
    def play_game_vs_stockfish(self, bot_is_white=True, stockfish_time_limit=0.1, starting_board=None, show_debug=False):
        board = starting_board.copy() if starting_board else chess.Board()
        game_positions = []
        max_moves = 400
        move_count = 0

        mcts = MCTS(self.network, num_simulations=self.num_simulations,
                    device=self.device, eval_batch_size=self.eval_batch_size, debug=self.mcts_debug)

        while not board.is_game_over() and move_count < max_moves:
            is_bot_turn = (board.turn == chess.WHITE) == bot_is_white
            phase = classify_game_phase(board)

            # Stockfish evaluation for supervision
            stockfish_score = None
            if self.stockfish_engine:
                try:
                    info = self.stockfish_engine.analyse(board, chess.engine.Limit(time=0.02))
                    score = info.get("score")
                    cp = score.white().score(mate_score=100000)
                    stockfish_score = max(min(cp / 1000.0, 1.0), -1.0)
                    if show_debug or self.show_moves:
                        print(f"[DEBUG] Stockfish CP={cp} scaled={stockfish_score:.3f} fen={board.fen()}")
                except Exception as e:
                    stockfish_score = 0.0
                    if show_debug:
                        print(f"[DEBUG] Stockfish analyse failed: {e}")

            if is_bot_turn:
                # ensure no grad during search calls
                with torch.no_grad():
                    res = mcts.search(board)
                if isinstance(res, tuple):
                    root, selected_move = res
                else:
                    selected_move = res
                    root = MCTSNode(board)
                    try:
                        mcts._expand_node(root)
                    except Exception:
                        legal = list(board.legal_moves)
                        for m in legal:
                            child = MCTSNode(board.copy(), parent=root, move=m, prior_prob=1.0/len(legal))
                            root.children[m] = child

                # debug
                if show_debug or self.show_moves:
                    try:
                        top = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:8]
                        top_str = ", ".join([f"{board.san(mv)} v={ch.visit_count} p={ch.prior_prob:.3f} q={ch.value():.3f}" for mv, ch in top])
                        print(f"[DEBUG] MCTS root value={root.value():.4f} | top: {top_str}")
                    except Exception:
                        print(f"[DEBUG] MCTS root value={root.value():.4f}")

                # build move_probs for training
                move_probs = np.zeros(4096, dtype=np.float32)
                total_visits = sum(child.visit_count for child in root.children.values())
                if total_visits > 0:
                    tau = max(1.0, 2.0 - (self.iteration * 0.002))
                    denom = sum((c.visit_count ** (1.0 / tau)) for c in root.children.values())
                    for move, child in root.children.items():
                        idx = move.from_square * 64 + move.to_square
                        move_probs[idx] = (child.visit_count ** (1.0 / tau)) / (denom if denom > 0 else 1.0)
                else:
                    legal = list(board.legal_moves)
                    for m in legal:
                        idx = m.from_square * 64 + m.to_square
                        move_probs[idx] = 1.0 / len(legal)

                game_positions.append({
                    "board": board_to_tensor(board),
                    "move_probs": move_probs,
                    "turn": board.turn,
                    "phase": phase,
                    "stockfish_eval": stockfish_score,
                    "mcts_value": float(root.value()),
                    "final_outcome": None,
                    "td_target": None
                })

                if show_debug or self.show_moves:
                    try:
                        print(f"[DEBUG] Bot selects: {board.san(selected_move)}")
                    except Exception:
                        print(f"[DEBUG] Bot selects (uci): {selected_move}")

            else:
                try:
                    res = self.stockfish_engine.play(board, chess.engine.Limit(time=stockfish_time_limit))
                    selected_move = res.move
                    if show_debug:
                        try:
                            print(f"[DEBUG] Stockfish plays: {board.san(selected_move)}")
                        except Exception:
                            print(f"[DEBUG] Stockfish plays (uci): {selected_move}")
                except Exception as e:
                    selected_move = random.choice(list(board.legal_moves))
                    print(f"Stockfish play error: {e} -- picking random move {selected_move}")

            board.push(selected_move)
            move_count += 1

        # assign final outcomes & compute TD targets
        result = board.result()
        for pos in game_positions:
            if result == "1-0":
                pos["final_outcome"] = 1.0 if pos["turn"] else -1.0
            elif result == "0-1":
                pos["final_outcome"] = -1.0 if pos["turn"] else 1.0
            else:
                pos["final_outcome"] = 0.0

        gamma = 0.99
        next_value = 0.0
        for pos in reversed(game_positions):
            next_value = pos["final_outcome"] * 0.5 + 0.5 * pos["mcts_value"]
            pos["td_target"] = gamma * next_value + (1 - gamma) * pos.get("stockfish_eval", 0.0)

        if self.save_pgns or self.show_moves:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self.pgn_dir / f"sf_game_iter{self.iteration}_{timestamp}.pgn"
            if self.save_pgns:
                try:
                    self._save_pgn_from_board(board, result, filename)
                    if not self.fast_eval:
                        print(f"💾 Saved PGN: {filename}")
                except Exception as e:
                    print(f"⚠ Failed to save PGN: {e}")
            if self.show_moves and not self.fast_eval:
                print(f"Game result: {result} — moves:")
                self._print_moves_from_board(board)

        moves_summary = self.moves_one_line(board)
        return game_positions, result, move_count, moves_summary

    # -------------------------
    # Self-play (keeps behavior but with no_grad)
    # -------------------------
    def play_self_games(self, num_games=4, simulations_each=None, save_pgns=False, show_moves=False, show_debug=False):
        sim = simulations_each or self.num_simulations
        results = []
        summaries = []
        for gi in range(num_games):
            board = chess.Board()
            mcts_a = MCTS(self.network, num_simulations=sim, device=self.device, eval_batch_size=self.eval_batch_size, debug=self.mcts_debug)
            mcts_b = MCTS(self.network, num_simulations=sim, device=self.device, eval_batch_size=self.eval_batch_size, debug=self.mcts_debug)
            move_count = 0
            with torch.no_grad():
                while not board.is_game_over() and move_count < 200:
                    if board.turn == chess.WHITE:
                        res = mcts_a.search(board)
                    else:
                        res = mcts_b.search(board)
                    move = res[1] if isinstance(res, tuple) else res
                    board.push(move)
                    move_count += 1

            result = board.result()
            results.append(result)
            moves_summary = self.moves_one_line(board)
            summaries.append((moves_summary, result))
            if not self.fast_eval:
                print(f"Self-play game {gi+1}: Result {result}\n{moves_summary}\n")

            if save_pgns:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = self.pgn_dir / f"self_game_iter{self.iteration}_g{gi+1}_{timestamp}.pgn"
                try:
                    self._save_pgn_from_board(board, result, filename)
                except Exception as e:
                    print(f"⚠ Failed to save self-play PGN: {e}")
        return results, summaries

    # -------------------------
    # Training on batch
    # -------------------------
    def train_on_batch(self):
        if self.balance_phases:
            samples_per_phase = max(1, self.batch_size // 3)
            batch = []
            for phase, buffer in self.replay_buffer.items():
                if len(buffer) > 0:
                    sample_size = min(len(buffer), samples_per_phase)
                    batch.extend(random.sample(buffer, sample_size))
            if len(batch) == 0:
                return None
        else:
            if len(self.replay_buffer) == 0:
                return None
            sample_size = min(len(self.replay_buffer), self.batch_size)
            batch = random.sample(self.replay_buffer, sample_size)

        boards = torch.from_numpy(np.array([d['board'] for d in batch], dtype=np.float32)).to(self.device)
        target_probs = torch.from_numpy(np.array([d['move_probs'] for d in batch], dtype=np.float32)).to(self.device)

        # combined value target
        target_values_np = []
        for d in batch:
            td = d.get("td_target", 0.0)
            sf = d.get("stockfish_eval", 0.0)
            final = d.get("final_outcome", 0.0)
            combined = 0.5 * td + 0.25 * sf + 0.25 * final
            target_values_np.append([combined])
        target_values = torch.from_numpy(np.array(target_values_np, dtype=np.float32)).to(self.device)

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

    # -------------------------
    # Evaluate vs a snapshot (fast, cached, no_grad)
    # -------------------------
    def evaluate_vs_snapshot(self, snapshot_path=None, num_games=10, simulations_current=None, simulations_snapshot=None,
                             save_pgns=False, show_moves=False, show_debug=False):
        # resolve snapshot net (prefer cached)
        if self.initial_snapshot_net is not None:
            snapshot_net = self.initial_snapshot_net
        else:
            if not snapshot_path:
                print("⚠ No snapshot provided and no cached snapshot available")
                return None
            snapshot_path = Path(snapshot_path)
            if not snapshot_path.exists():
                print(f"⚠ Snapshot {snapshot_path} not found; skipping evaluation")
                return None
            snapshot_net = ChessNetwork().to(self.device)
            snapshot_net.load_state_dict(torch.load(snapshot_path, map_location=self.device))
            snapshot_net.eval()
            for p in snapshot_net.parameters():
                p.requires_grad = False

        sim_curr = simulations_current or self.num_simulations
        sim_prev = simulations_snapshot or max(self.snapshot_min_sims, int(sim_curr * self.snapshot_sim_fraction))

        # print summary (only brief)
        print(f"  🎮 NEW model: {sim_curr} sims | OLD (snapshot) model: {sim_prev} sims | games: {num_games}")

        wins = draws = losses = 0.0
        moves_list = []

        # ensure nets in eval and no grad
        self.network.eval()
        snapshot_net.eval()
        for p in snapshot_net.parameters(): p.requires_grad = False

        # single-threaded default; optional parallelization (use with caution on GPU)
        if self.parallel_eval_workers > 0 and self.is_cpu:
            # Parallel evaluation strategy (CPU-only recommended)
            results = self._parallel_evaluate_games(snapshot_net, num_games, sim_curr, sim_prev, save_pgns, show_moves, show_debug)
            for res, move_count in results:
                moves_list.append(move_count)
                if res == "1-0":
                    wins += 1
                elif res == "0-1":
                    losses += 1
                else:
                    draws += 1
        else:
            # sequential evaluation
            for i in range(num_games):
                board = chess.Board()
                move_count = 0
                max_moves = 400

                mcts_curr = MCTS(self.network, num_simulations=sim_curr, device=self.device, eval_batch_size=self.eval_batch_size, debug=self.mcts_debug)
                mcts_prev = MCTS(snapshot_net, num_simulations=sim_prev, device=self.device, eval_batch_size=self.eval_batch_size, debug=self.mcts_debug)

                curr_is_white = (i % 2 == 0)

                # disable autograd during the whole game to avoid overhead
                with torch.no_grad():
                    while not board.is_game_over() and move_count < max_moves:
                        if (board.turn == chess.WHITE) == curr_is_white:
                            res = mcts_curr.search(board)
                            move = res[1] if isinstance(res, tuple) else res
                        else:
                            res = mcts_prev.search(board)
                            move = res[1] if isinstance(res, tuple) else res
                        board.push(move)
                        move_count += 1

                res = board.result()
                moves_list.append(move_count)

                if save_pgns and not self.fast_eval:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = self.pgn_dir / f"eval_iter{self.iteration}_game{i+1}_{timestamp}.pgn"
                        self._save_pgn_from_board(board, res, filename)
                    except Exception as e:
                        print(f"⚠ Failed to save eval PGN: {e}")

                if not self.fast_eval:
                    print(f"Eval game {i+1}: {res} in {move_count} moves")
                    if show_moves:
                        self._print_moves_from_board(board)

                if res == "1-0":
                    # if curr_is_white is True -> new model as white
                    wins += 1 if curr_is_white else 0
                    losses += 0 if curr_is_white else 1
                elif res == "0-1":
                    losses += 1 if curr_is_white else 0
                    wins += 0 if curr_is_white else 1
                else:
                    draws += 1

        total = num_games
        winrate = (wins + 0.5 * draws) / total if total > 0 else 0.0
        stats = {
            "games": total,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "winrate": winrate,
            "avg_moves": float(np.mean(moves_list)) if moves_list else 0.0
        }
        return stats

    def _parallel_evaluate_games(self, snapshot_net, num_games, sim_curr, sim_prev, save_pgns, show_moves, show_debug):
        """
        Helper to evaluate games in parallel using multiprocessing Pool.
        NOTE: For GPU use, avoid parallel GPU workers unless you know how to manage device contexts.
        This routine is only recommended on CPU or when you have per-process GPU assignment.
        """
        # Worker needs a picklable callable; pass minimal information like model paths, not model objects
        # For simplicity, this implementation is CPU-only: we re-load snapshot and current network state_dict
        curr_state = self.network.state_dict()
        tmp_snapshot_path = None
        results = []

        def worker(args):
            idx, curr_sd, snapshot_sd, device_str, simc, simp = args
            # lightweight worker that reconstructs nets (CPU) and runs a game
            dev = torch.device(device_str)
            curr_net = ChessNetwork().to(dev)
            curr_net.load_state_dict(curr_sd)
            curr_net.eval()
            snap_net = ChessNetwork().to(dev)
            snap_net.load_state_dict(snapshot_sd)
            snap_net.eval()
            for p in snap_net.parameters(): p.requires_grad = False

            mcts_curr = MCTS(curr_net, num_simulations=simc, device=dev, eval_batch_size=self.eval_batch_size, debug=False)
            mcts_prev = MCTS(snap_net, num_simulations=simp, device=dev, eval_batch_size=self.eval_batch_size, debug=False)

            board = chess.Board()
            move_count = 0
            with torch.no_grad():
                curr_is_white = (idx % 2 == 0)
                while not board.is_game_over() and move_count < 400:
                    if (board.turn == chess.WHITE) == curr_is_white:
                        res = mcts_curr.search(board)
                        mv = res[1] if isinstance(res, tuple) else res
                    else:
                        res = mcts_prev.search(board)
                        mv = res[1] if isinstance(res, tuple) else res
                    board.push(mv)
                    move_count += 1
            return board.result(), move_count

        # prepare args
        snap_sd = {k: v.cpu() for k, v in (self.initial_snapshot_net.state_dict().items() if self.initial_snapshot_net else torch.load(self.initial_snapshot_path, map_location='cpu').items())}
        curr_sd = {k: v.cpu() for k, v in curr_state.items()}

        args_list = []
        for i in range(num_games):
            args_list.append((i, curr_sd, snap_sd, 'cpu', sim_curr, sim_prev))

        with mp.Pool(processes=min(self.parallel_eval_workers, mp.cpu_count())) as pool:
            results = pool.map(worker, args_list)
        return results

    # -------------------------
    # Main loop: three phases per cycle (optimized)
    # -------------------------
    def run_for_minutes(self,
                        minutes_limit=30,
                        sf_games=100,
                        selfplay_games=10,
                        first_vs_games=4,
                        training_batches=40,
                        stockfish_skill=1,
                        stockfish_time_limit=0.02,
                        save_every=1,
                        varied_starts=True):
        end_time = time.time() + (minutes_limit * 60)
        cycle = 0

        self.stockfish_engine = init_stockfish(self.stockfish_path, skill_level=stockfish_skill, threads=self.stockfish_threads)
        if not self.stockfish_engine:
            print("⚠ Stockfish not available — exiting.")
            return

        try:
            while time.time() < end_time:
                cycle += 1
                self.iteration += 1
                print("\n" + "=" * 60)
                print(f"🔁 Cycle {cycle} (iteration {self.iteration})")
                print("=" * 60)

                # Save pre-cycle snapshot
                pre_snapshot = self._save_model_snapshot(snapshot_name=f"snapshot_pre_iter_{self.iteration}.pth")

                # PHASE A: vs Stockfish
                print(f"\n⚔️ Phase A: Playing {sf_games} games vs Stockfish")
                wins = 0.0
                phase_counts = defaultdict(int)

                for g in range(sf_games):
                    bot_white = (g % 2 == 0)
                    if varied_starts and g > 0:
                        target_phase = ["opening", "midgame", "endgame"][g % 3]
                        starting_board = generate_varied_starting_position(self.stockfish_engine, target_phase)
                        if not self.fast_eval:
                            print(f"  Game {g+1} starting from {target_phase} position")
                    else:
                        starting_board = None

                    positions, result, moves, moves_summary = self.play_game_vs_stockfish(
                        bot_is_white=bot_white,
                        stockfish_time_limit=stockfish_time_limit,
                        starting_board=starting_board,
                        show_debug=self.mcts_debug
                    )
                    if not self.fast_eval:
                        print(f"Game {g+1} Result: {result} - Moves: {moves}\n{moves_summary}\n")

                    # add to replay buffer
                    if self.balance_phases:
                        for pos in positions:
                            phase = pos.get("phase", "midgame")
                            self.replay_buffer[phase].append(pos)
                            phase_counts[phase] += 1
                    else:
                        self.replay_buffer.extend(positions)

                    self.stats['total_games'] += 1
                    if (result == "1-0" and bot_white) or (result == "0-1" and not bot_white):
                        wins += 1
                    elif result == "1/2-1/2":
                        wins += 0.5

                    if not self.fast_eval:
                        print(f"  Result: {result} in {moves} moves - {len(positions)} positions")

                if self.balance_phases and not self.fast_eval:
                    print(f"\n  📊 Phase distribution after Stockfish phase:")
                    for phase in ["opening", "midgame", "endgame"]:
                        count = phase_counts[phase]
                        buffer_size = len(self.replay_buffer[phase])
                        print(f"    {phase.capitalize()}: {count} new | {buffer_size} total in buffer")

                winrate_sf = wins / sf_games if sf_games > 0 else 0
                self.stats['phase1_winrates'].append(winrate_sf)
                if not self.fast_eval:
                    print(f"\n  🎯 PhaseA winrate (bot vs SF): {winrate_sf*100:.1f}%")

                # Train after Stockfish
                if not self.fast_eval:
                    print(f"\n📚 Training: {training_batches} batches (after Stockfish)")
                losses = []
                for b in range(training_batches):
                    loss = self.train_on_batch()
                    if loss:
                        losses.append(loss['total_loss'])
                    if not self.fast_eval and loss:
                        print(f"    Batch {b+1}/{training_batches} - Total Loss: {loss['total_loss']:.4f} "
                              f"(Policy: {loss['policy_loss']:.4f}, Value: {loss['value_loss']:.4f})")

                if losses:
                    avg_loss = float(np.mean(losses))
                    self.stats['recent_losses'].append(avg_loss)
                    if len(self.stats['recent_losses']) > 10:
                        self.stats['recent_losses'].pop(0)
                    self.scheduler.step(avg_loss)
                    if not self.fast_eval:
                        print(f"  📈 Avg training loss (after SF): {avg_loss:.4f}")

                # PHASE B: Self-play
                if not self.fast_eval:
                    print(f"\n🤝 Phase B: Self-play {selfplay_games} games")
                self.play_self_games(num_games=selfplay_games, simulations_each=2 if self.is_cpu else None,
                                     save_pgns=self.save_pgns, show_moves=self.show_moves)

                # Train after self-play
                if not self.fast_eval:
                    print(f"\n📚 Training: {training_batches} batches (after self-play)")
                losses = []
                for b in range(training_batches):
                    loss = self.train_on_batch()
                    if loss:
                        losses.append(loss['total_loss'])
                if losses:
                    avg_loss = float(np.mean(losses))
                    self.stats['recent_losses'].append(avg_loss)
                    if len(self.stats['recent_losses']) > 10:
                        self.stats['recent_losses'].pop(0)
                    self.scheduler.step(avg_loss)
                    if not self.fast_eval:
                        print(f"  📈 Avg training loss (after self-play): {avg_loss:.4f}")

                # PHASE C: vs initial (first-version)
                print(f"\n🥊 Phase C: Evaluation vs FIRST version ({self.initial_snapshot_path.name}) - {first_vs_games} games")
                eval_stats = self.evaluate_vs_snapshot(
                    str(self.initial_snapshot_path),
                    num_games=first_vs_games,
                    simulations_current=2 if self.is_cpu else self.num_simulations,
                    simulations_snapshot=2 if self.is_cpu else max(self.snapshot_min_sims, int(self.num_simulations * self.snapshot_sim_fraction)),
                    save_pgns=self.save_pgns and (not self.fast_eval),
                    show_moves=self.show_moves and (not self.fast_eval),
                    show_debug=self.mcts_debug
                )

                if eval_stats and not self.fast_eval:
                    print(f"\n  🏆 Winrate vs first-version: {eval_stats['winrate']*100:.1f}%")

                # periodic checkpoint
                if (self.iteration % save_every) == 0:
                    checkpoint = {
                        'iteration': self.iteration,
                        'network_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'stats': self.stats
                    }
                    ck_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
                    torch.save(checkpoint, ck_path)
                    print(f"💾 Checkpoint saved: {ck_path}")

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")

        finally:
            if self.stockfish_engine:
                self.stockfish_engine.quit()
            try:
                torch.save(self.network.state_dict(), MODEL_PATH)
                print(f"✅ Final model saved to {MODEL_PATH}")
            except Exception:
                print("⚠ Could not save final model to MODEL_PATH")

    # -------------------------
    # IO: save canonical snapshot
    # -------------------------
    def _save_model_snapshot(self, snapshot_name=None):
        if snapshot_name is None:
            snapshot_name = f"snapshot_iter_{self.iteration}.pth"
        snapshot_path = self.checkpoint_dir / snapshot_name
        torch.save(self.network.state_dict(), snapshot_path)
        try:
            torch.save(self.network.state_dict(), MODEL_PATH)
        except Exception:
            pass
        return snapshot_path

    def _load_latest_model_if_exists(self):
        main_path = Path(MODEL_PATH)
        if main_path.exists():
            try:
                sd = torch.load(main_path, map_location=self.device)
                self.network.load_state_dict(sd)
                if hasattr(self.network, "enable_fast_inference"):
                    try:
                        self.network.enable_fast_inference()
                    except Exception:
                        pass
                print(f"✓ Loaded main model from {main_path}")
            except Exception as e:
                print(f"⚠ Could not load model: {e}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--sf-games", type=int, default=100, help="Games vs Stockfish per cycle (reduced default)")
    parser.add_argument("--selfplay-games", type=int, default=6, help="Self-play games per cycle")
    parser.add_argument("--first-vs-games", type=int, default=4, help="Games vs initial first-version per cycle")
    parser.add_argument("--training-batches", type=int, default=128)
    parser.add_argument("--stockfish-path", type=str, default="dist/stockfish-ubuntu-x86-64-avx2")
    parser.add_argument("--stockfish-skill", type=int, default=1)
    parser.add_argument("--stockfish-time", type=float, default=0.02)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--snapshot-sim-fraction", type=float, default=0.25, help="Fraction of simulations for snapshot side")
    parser.add_argument("--snapshot-min-sims", type=int, default=8, help="Min sims for snapshot side")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--no-balance", action="store_true", help="Disable phase balancing")
    parser.add_argument("--no-varied-starts", action="store_true", help="Always start from initial position")
    parser.add_argument("--show-moves", action="store_true", help="Print moves as games are played")
    parser.add_argument("--save-pgns", action="store_true", help="Save PGNs for games (phase1/self/eval)")
    parser.add_argument("--pgn-dir", type=str, default="games")
    parser.add_argument("--self-games", type=int, default=0, help="Number of self-play games to run/print each cycle")
    parser.add_argument("--mcts-debug", action="store_true", help="Enable MCTS debug prints")
    parser.add_argument("--stockfish-threads", type=int, default=None, help="Threads to give Stockfish")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="MCTS eval batch size (bigger batches -> faster GPU)")
    parser.add_argument("--fast-eval", action="store_true", help="Silence heavy IO during Phase C for speed")
    parser.add_argument("--parallel-eval-workers", type=int, default=0, help="Parallel evaluation workers (CPU-only recommended)")
    args = parser.parse_args()

    trainer = EnhancedStockfishTrainer(
        stockfish_path=args.stockfish_path,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_simulations=args.simulations,
        snapshot_sim_fraction=args.snapshot_sim_fraction,
        snapshot_min_sims=args.snapshot_min_sims,
        device=None,
        checkpoint_dir="checkpoints",
        torch_threads=args.torch_threads,
        balance_phases=not args.no_balance,
        show_moves=args.show_moves,
        save_pgns=args.save_pgns,
        pgn_dir=args.pgn_dir,
        self_games_per_cycle=args.self_games,
        mcts_debug=args.mcts_debug,
        stockfish_threads=args.stockfish_threads,
        eval_batch_size=args.eval_batch_size,
        fast_eval=args.fast_eval,
        parallel_eval_workers=args.parallel_eval_workers
    )

    trainer.run_for_minutes(
        minutes_limit=args.minutes,
        sf_games=args.sf_games,
        selfplay_games=args.selfplay_games,
        first_vs_games=args.first_vs_games,
        training_batches=args.training_batches,
        stockfish_skill=args.stockfish_skill,
        stockfish_time_limit=args.stockfish_time,
        varied_starts=not args.no_varied_starts
    )

if __name__ == "__main__":
    main()
