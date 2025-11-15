import os
import random
import chess.pgn
import numpy as np
import torch
from torch import nn, optim
from collections import defaultdict
from board_state import board_to_tensor
from network_architecture import ChessNetwork

# -------------------------------
# Game Phase Classification
# -------------------------------
def classify_game_phase(board):
    move_count = board.fullmove_number
    piece_count = len(board.piece_map())
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))

    if piece_count <= 10 or (queens == 0 and rooks <= 2):
        return "endgame"
    elif move_count <= 12 and piece_count >= 28:
        return "opening"
    else:
        return "midgame"


def is_position_worth_learning(board, move_number):
    if move_number < 5:
        return False
    if len(list(board.legal_moves)) < 3:
        return False
    if board.is_checkmate() or board.is_stalemate():
        return False
    return True


# -------------------------------
# Load PGN games with filtering
# -------------------------------
def load_pgn_games(pgn_folder, min_elo=1800, max_games_per_file=1000):
    games = []
    for filename in os.listdir(pgn_folder):
        if not filename.endswith(".pgn"):
            continue
        path = os.path.join(pgn_folder, filename)
        loaded = 0
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                    avg_elo = (white_elo + black_elo) / 2
                    if avg_elo >= min_elo:
                        games.append(game)
                        loaded += 1
                except Exception:
                    # skip games missing ELO header or malformed
                    pass
                if loaded >= max_games_per_file:
                    break
        print(f"✓ Finished {filename}: loaded {loaded} games (avg ELO >= {min_elo})")
    return games


# -------------------------------
# Prepare training data
# -------------------------------
def prepare_training_data(games,
                          focus_phase="all",
                          opening_weight=0.3,
                          midgame_weight=0.4,
                          endgame_weight=0.3,
                          skip_early_moves=5):
    training_data = {"opening": [], "midgame": [], "endgame": []}

    for j, game in enumerate(games, 1):
        board = chess.Board()
        outcome = game.headers.get("Result", "*")
        if outcome not in ["1-0", "0-1", "1/2-1/2"]:
            continue

        moves_list = list(game.mainline_moves())
        total_moves = len(moves_list)
        if total_moves < 15:
            continue

        for move_number, move in enumerate(moves_list, start=1):
            if move_number <= skip_early_moves:
                board.push(move)
                continue

            if not is_position_worth_learning(board, move_number):
                board.push(move)
                continue

            phase = classify_game_phase(board)

            # discounted value target
            moves_to_end = total_moves - move_number
            discount_factor = np.exp(-moves_to_end / 20.0)
            if outcome == "1-0":
                base_value = 1.0 if board.turn == chess.WHITE else -1.0
            elif outcome == "0-1":
                base_value = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                base_value = 0.0
            value_target = base_value * discount_factor

            # position tensor (ensure numpy float32)
            position_array = np.array(board_to_tensor(board), dtype=np.float32)

            # policy target: uniform over legal moves, slight boost for played move
            policy_target = np.zeros(64 * 64, dtype=np.float32)
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                board.push(move)
                continue

            for legal_move in legal_moves:
                idx = legal_move.from_square * 64 + legal_move.to_square
                policy_target[idx] = 1.0 / len(legal_moves)

            # boost actual move slightly
            actual_idx = move.from_square * 64 + move.to_square
            policy_target[actual_idx] += 0.3
            # normalize
            policy_target = policy_target / policy_target.sum()

            training_data[phase].append({
                "position": position_array,
                "policy_target": policy_target,
                "value_target": float(value_target),
                "move_number": move_number,
                "phase": phase
            })

            board.push(move)

        if j % 50 == 0:
            print(f"Processed {j} games (collected so far: "
                  f"open={len(training_data['opening'])}, "
                  f"mid={len(training_data['midgame'])}, "
                  f"end={len(training_data['endgame'])})")

    # print stats
    print("\n" + "="*60)
    print("📊 Training Data Statistics:")
    for phase in ["opening", "midgame", "endgame"]:
        print(f"{phase.capitalize()}: {len(training_data[phase]):,} positions")
    print("="*60 + "\n")

    # combine based on weights if focus == all
    if focus_phase == "all":
        combined = []
        total_positions = sum(len(training_data[p]) for p in training_data)
        weights = {"opening": opening_weight, "midgame": midgame_weight, "endgame": endgame_weight}
        for phase, weight in weights.items():
            phase_data = training_data[phase]
            if len(phase_data) == 0:
                continue
            target_samples = int(total_positions * weight)
            if target_samples <= 0:
                continue
            if len(phase_data) < target_samples:
                idxs = np.random.choice(len(phase_data), target_samples, replace=True)
            else:
                idxs = np.random.choice(len(phase_data), target_samples, replace=False)
            combined.extend([phase_data[i] for i in idxs])
        random.shuffle(combined)
        return combined

    elif focus_phase in training_data:
        return training_data[focus_phase]

    else:
        # fallback to all combined
        combined = training_data["opening"] + training_data["midgame"] + training_data["endgame"]
        random.shuffle(combined)
        return combined


# -------------------------------
# Main training
# -------------------------------
def train(pgn_folder="./pgn_folder",
          focus_phase="all",
          opening_weight=0.2,
          midgame_weight=0.4,
          endgame_weight=0.4,
          epochs=5,
          batch_size=64,
          learning_rate=1e-4,
          model_path="models/trained_model.pth",
          entropy_coef=0.0,
          device_override=None):
    device = torch.device(device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("🔧 Using device:", device)
    if device.type == "cuda":
        print("   GPU:", torch.cuda.get_device_name(0))

    print("\n📖 Loading PGN games...")
    games = load_pgn_games(pgn_folder, min_elo=1800, max_games_per_file=1000)
    print(f"✓ Total quality games loaded: {len(games)}\n")

    if len(games) == 0:
        print("❌ No games loaded! Check your PGN folder.")
        return

    print(f"⚙️  Preparing training data (focus: {focus_phase})...")
    training_data = prepare_training_data(games,
                                          focus_phase=focus_phase,
                                          opening_weight=opening_weight,
                                          midgame_weight=midgame_weight,
                                          endgame_weight=endgame_weight)
    print(f"✓ Total training positions: {len(training_data):,}\n")

    if len(training_data) == 0:
        print("❌ No training data prepared!")
        return

    # model and optimizer
    network = ChessNetwork().to(device)
    if os.path.exists(model_path):
        try:
            network.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}\n   Starting with fresh weights")

    network.train()
    optimizer = optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # losses
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # expects log-probs vs probs
    value_loss_fn = nn.SmoothL1Loss()  # Huber-style stable regression loss

    print("=" * 60)
    print("🚀 Starting Training")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        random.shuffle(training_data)
        epoch_losses = []
        policy_losses = []
        value_losses = []

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            # safety: ensure shapes are consistent
            positions_np = np.stack([np.array(d["position"], dtype=np.float32) for d in batch])
            policy_np = np.stack([np.array(d["policy_target"], dtype=np.float32) for d in batch])
            value_np = np.array([[d["value_target"]] for d in batch], dtype=np.float32)

            positions = torch.from_numpy(positions_np).to(device)
            policy_targets = torch.from_numpy(policy_np).to(device)
            value_targets = torch.from_numpy(value_np).to(device)

            # forward
            policy_pred_logits, value_pred = network(positions)  # expect (B, P) and (B, 1) shapes

            # policy loss (KLDiv requires log-probs input)
            log_probs = torch.log_softmax(policy_pred_logits, dim=1)
            policy_loss = policy_loss_fn(log_probs, policy_targets)

            # value loss
            value_loss = value_loss_fn(value_pred, value_targets)

            # optional entropy bonus to encourage exploration (usually small)
            if entropy_coef and entropy_coef > 0.0:
                probs = torch.softmax(policy_pred_logits, dim=1)
                entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
            else:
                entropy = 0.0

            # combined loss (weights can be tuned; 1.0/1.0 is standard for AlphaZero; we keep adjustable mix)
            total_loss = 0.7 * policy_loss + 0.3 * value_loss - entropy_coef * (entropy if isinstance(entropy, torch.Tensor) else 0.0)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # progress print every N batches
            batch_num = i // batch_size + 1
            if batch_num % 50 == 0 or (i + batch_size) >= len(training_data):
                recent = np.mean(epoch_losses[-50:]) if len(epoch_losses) >= 1 else float('nan')
                print(f"Epoch {epoch}/{epochs} | Batch {batch_num} / {int(np.ceil(len(training_data)/batch_size))} "
                      f"| loss: {recent:.4f} (policy: {np.mean(policy_losses[-50:]):.4f}, value: {np.mean(value_losses[-50:]):.4f})")

        avg_epoch_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else 0.0
        avg_policy = float(np.mean(policy_losses)) if len(policy_losses) else 0.0
        avg_value = float(np.mean(value_losses)) if len(value_losses) else 0.0

        print(f"\n✓ Epoch {epoch}/{epochs} completed — avg loss: {avg_epoch_loss:.4f} "
              f"(policy: {avg_policy:.4f}, value: {avg_value:.4f})\n")

        scheduler.step(avg_epoch_loss)

    # final save
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(network.state_dict(), model_path)
    print("=" * 60)
    print(f"✅ Model saved to {model_path}")
    print("=" * 60)


# -------------------------------
# CLI entrypoint
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train chess network from PGN games")
    parser.add_argument("--pgn-folder", type=str, default="./pgn_folder")
    parser.add_argument("--focus", type=str, default="all", choices=["all", "opening", "midgame", "endgame"])
    parser.add_argument("--opening-weight", type=float, default=0.2)
    parser.add_argument("--midgame-weight", type=float, default=0.4)
    parser.add_argument("--endgame-weight", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-path", type=str, default="models/midgame_model.pth")
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    args = parser.parse_args()

    train(pgn_folder=args.pgn_folder,
          focus_phase=args.focus,
          opening_weight=args.opening_weight,
          midgame_weight=args.midgame_weight,
          endgame_weight=args.endgame_weight,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          model_path=args.model_path,
          entropy_coef=args.entropy_coef)
