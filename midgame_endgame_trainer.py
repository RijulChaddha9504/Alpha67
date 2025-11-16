#!/usr/bin/env python3
import os
import chess
import chess.pgn
import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from chess.engine import SimpleEngine, Limit
import argparse

from board_state import board_to_tensor
from network_architecture import ChessNetwork


###########################################
# CONFIG
###########################################
STOCKFISH_PATH = "dist/stockfish-ubuntu-x86-64-avx2"

BASE_DEPTH = 6          # adaptive depth (will increase for endgames)
CP_SCALE = 300
USE_MIXED_PRECISION = True
CACHE_SIZE = 50000      # Stockfish eval cache size (LRU)


###########################################
# LRU cache for Stockfish evaluations
###########################################
_eval_cache = OrderedDict()


def _cache_get(fen):
    return _eval_cache.get(fen, None)


def _cache_set(fen, val):
    _eval_cache[fen] = val
    # evict oldest
    if len(_eval_cache) > CACHE_SIZE:
        _eval_cache.popitem(last=False)


###########################################
# STOCKFISH EVALUATION (with cache)
###########################################
def evaluate_with_stockfish(engine, board):
    """Return normalized value from -1 to 1 using Stockfish evaluation with LRU caching."""
    fen = board.fen()
    cached = _cache_get(fen)
    if cached is not None:
        return cached

    # Adaptive depth: deeper when fewer pieces remain
    piece_count = len(board.piece_map())
    depth = BASE_DEPTH if piece_count >= 20 else BASE_DEPTH + 3

    val = 0.0
    try:
        info = engine.analyse(board, Limit(depth=depth))
        score = info.get("score")
        if score is None:
            val = 0.0
        else:
            score = score.white()
            if score.is_mate():
                mate = score.mate()
                if mate is None:
                    val = 0.0
                else:
                    val = 1.0 if mate > 0 else -1.0
            else:
                cp = score.score()
                if cp is None:
                    val = 0.0
                else:
                    val = float(np.tanh(cp / CP_SCALE))
    except Exception:
        val = 0.0

    _cache_set(fen, val)
    return val


###########################################
# GAME PHASE LOGIC
###########################################
def classify_game_phase(board):
    piece_count = len(board.piece_map())
    if piece_count <= 14:
        return "endgame"
    elif piece_count >= 26:
        return "opening"
    return "midgame"


###########################################
# PGN LOADER (fast)
###########################################
def load_pgn_games(pgn_folder, max_games_per_file=5000):
    games = []
    if not os.path.isdir(pgn_folder):
        print(f"❌ PGN folder not found: {pgn_folder}")
        return games

    for filename in sorted(os.listdir(pgn_folder)):
        if not filename.endswith(".pgn"):
            continue
        path = os.path.join(pgn_folder, filename)
        loaded = 0
        with open(path, encoding="utf-8", errors="ignore") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.append(g)
                loaded += 1
                if loaded >= max_games_per_file:
                    break
        print(f"   ✓ Loaded {loaded} games from {filename}")
    return games


###########################################
# DATASET BUILDER
###########################################
def prepare_data(games, engine, debug=False):
    phases = defaultdict(list)

    for game in tqdm(games, desc="Extracting"):
        board = chess.Board()
        moves = list(game.mainline_moves())
        for move in moves:
            # safety: if no legal moves for some reason, skip
            legal = list(board.legal_moves)
            if len(legal) == 0:
                board.push(move)
                continue

            pos = board_to_tensor(board)
            phase = classify_game_phase(board)

            # policy vector (vectorized-ish)
            policy = np.zeros(4096, dtype=np.float32)
            inv = 1.0 / len(legal)
            for m in legal:
                idx = m.from_square * 64 + m.to_square
                policy[idx] = inv

            teacher_idx = move.from_square * 64 + move.to_square
            policy[teacher_idx] += 0.25
            # re-normalize
            s = policy.sum()
            if s > 0:
                policy /= s

            value = evaluate_with_stockfish(engine, board)

            phases[phase].append((pos, policy, value))
            board.push(move)

    print("Phase counts:", {k: len(v) for k, v in phases.items()})

    # If any phase has zero samples, avoid sampling error
    for p in ("opening", "midgame", "endgame"):
        if p not in phases or len(phases[p]) == 0:
            phases[p] = []

    # Select counts per phase (you can tune these)
    target_open = 20000
    target_mid = 40000
    target_end = 60000

    # If debug: we only want a tiny dataset for quick checks
    if debug:
        # just pick up to 100 combined samples (prefer endgame)
        selected = []
        # try to pull from endgame first
        for src in ("endgame", "midgame", "opening"):
            take = min(100 - len(selected), len(phases[src]))
            if take > 0:
                idxs = np.random.choice(len(phases[src]), take, replace=False)
                selected.extend([phases[src][i] for i in idxs])
            if len(selected) >= 100:
                break
        print(f"DEBUG mode: prepared {len(selected)} samples (should be 100 or fewer)")
        return selected

    # Normal (non-debug) flow: sample indices (with replacement if needed)
    final = []

    def sample_from_phase(phase_list, count):
        if len(phase_list) == 0:
            return []
        if len(phase_list) >= count:
            idxs = np.random.choice(len(phase_list), count, replace=False)
        else:
            idxs = np.random.choice(len(phase_list), count, replace=True)
        return [phase_list[i] for i in idxs]

    final.extend(sample_from_phase(phases["opening"], target_open))
    final.extend(sample_from_phase(phases["midgame"], target_mid))
    final.extend(sample_from_phase(phases["endgame"], target_end))

    np.random.shuffle(final)
    print("Prepared final dataset size:", len(final))
    return final


###########################################
# TRAINING LOOP (with debug flag)
###########################################
def train(model_path="models/fast_model.pth", pgn_folder="pgn_folder", debug=False):
    # start engine
    engine = SimpleEngine.popen_uci(STOCKFISH_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Device: {device}")

    # model init / load
    model = ChessNetwork().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("🔁 Loaded existing model weights.")
        except Exception as e:
            print("⚠ Could not load model weights:", e)

    # optional: compile with TorchScript (keeps interface same)
    try:
        scripted = torch.jit.script(model)
        model = scripted
        print("⚡ Model compiled with TorchScript.")
    except Exception as e:
        print("⚠ TorchScript compilation failed, continuing with the Python model:", e)

    # optimizer & scaler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=(USE_MIXED_PRECISION and torch.cuda.is_available()))

    # load games
    games = load_pgn_games(pgn_folder)

    # ---------------------------
    # DEBUG GAME LIMIT: if debug, KEEP ONLY THE FIRST 50 GAMES
    # ---------------------------
    if debug:
        original_count = len(games)
        games = games[:50]
        print(f"🔧 DEBUG MODE → Limiting games to first {len(games)} of {original_count} total loaded.")
    # ---------------------------

    # prepare data from those games
    data = prepare_data(games, engine, debug=debug)

    # Debug: show a sample entry
    if debug and len(data) > 0:
        print("Sample entry (position shape, policy nonzero count, value):",
              np.array(data[0][0]).shape,
              int(np.count_nonzero(data[0][1])),
              data[0][2])

    # if debug reduce batch size and epochs for speed
    epochs = 4 if debug else 4
    batch_size = 8 if debug else 128

    print(f"Starting training | epochs={epochs} batch_size={batch_size} samples={len(data)}")

    for epoch in range(epochs):
        epoch_losses = []
        for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}"):
            batch = data[i:i+batch_size]
            if len(batch) == 0:
                continue

            positions = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(device, non_blocking=True)
            policy_targets = torch.tensor([b[1] for b in batch], dtype=torch.float32).to(device)
            value_targets = torch.tensor([[b[2]] for b in batch], dtype=torch.float32).to(device)

            with autocast(enabled=(USE_MIXED_PRECISION and torch.cuda.is_available())):
                policy_pred, value_pred = model(positions)
                policy_loss = -torch.mean(torch.sum(policy_targets * torch.log_softmax(policy_pred, dim=1), dim=1))
                value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)
                loss = policy_loss * 0.6 + value_loss * 0.4
                if debug and (i // batch_size) % 50 == 0:
                    # Show first entry of batch
                    sf_score = value_targets[0].item()
                    pred_value = value_pred[0].item()

                    # Extract a top predicted move index
                    best_move_idx = torch.argmax(policy_pred[0]).item()
                    
                    print("\n---------------- DEBUG SAMPLE ----------------")
                    print(f"Batch step: {i // batch_size}")
                    print(f"Predicted Value:     {pred_value:.4f}")
                    print(f"Stockfish Target:    {sf_score:.4f}")
                    print(f"Value Loss:          {value_loss.item():.4f}")
                    print(f"Policy Loss:         {policy_loss.item():.4f}")
                    print(f"Total Loss:          {loss.item():.4f}")
                    print(f"Top Policy Move ID:  {best_move_idx}")
                    print("------------------------------------------------")
                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"📉 Epoch {epoch+1} loss: {avg:.4f}")

    # final save
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved → {model_path}")

    engine.quit()


###########################################
# CLI
###########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn-folder", type=str, default="pgn_folder")
    parser.add_argument("--model-path", type=str, default="models/fast_model.pth")
    parser.add_argument("--debug", action="store_true", help="Run small debug data pass (process only first 50 games)")
    args = parser.parse_args()

    train(model_path=args.model_path, pgn_folder=args.pgn_folder, debug=args.debug)
