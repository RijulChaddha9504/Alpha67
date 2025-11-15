import os
import chess.pgn
import numpy as np
import torch
from torch import nn, optim
from board_state import board_to_tensor
from network_architecture import ChessNetwork

# Training model file
# -------------------------------
# 1️⃣ Load PGN games
# -------------------------------
def load_pgn_games(pgn_folder):
    games = []
    for filename in os.listdir(pgn_folder):
        if filename.endswith(".pgn"):
            path = os.path.join(pgn_folder, filename)
            with open(path) as f:
                i = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    games.append(game)
                    i += 1
                    if (i == 1000):  # Limit for speed
                        break

                    if i % 100 == 0:
                        print(f"Loaded {i} games from {filename}")
                print(f"Finished loading {filename}, total {i} games")
    return games

# -------------------------------
# 2️⃣ Prepare training data
# -------------------------------
def prepare_training_data(games, max_games=1000):
    # games = games[:max_games]  # Limit for speed
    training_data = []

    for j, game in enumerate(games, 1):
        board = chess.Board()
        outcome = game.headers["Result"]
        value = 1 if outcome == "1-0" else -1 if outcome == "0-1" else 0

        for move in game.mainline_moves():
            position_tensor = board_to_tensor(board)

            policy_target = np.full(64*64, 1e-4, dtype=np.float32)
            idx = move.from_square * 64 + move.to_square
            policy_target[idx] = 1.0 - (64*64 - 1) * 1e-4

            training_data.append({
                "position": position_tensor,
                "policy_target": policy_target,
                "value_target": value
            })

            board.push(move)

        if j % 50 == 0:
            print(f"Prepared training data from {j} games")

    return training_data

# -------------------------------
# 3️⃣ Main training
# -------------------------------
def train():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # Load games
    games = load_pgn_games("./pgn_folder")
    print(f"Total games loaded: {len(games)}")

    # Prepare data
    training_data = prepare_training_data(games)
    print(f"Total training positions: {len(training_data)}")

    # Initialize network and optimizer
    network = ChessNetwork().to(device)
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Training parameters
    epochs = 5
    batch_size = 64

    # Training loop
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            # Prepare batch tensors and move to device
            positions = torch.tensor(
                np.array([d["position"] for d in batch], dtype=np.float32),
                device=device
            )
            policy_targets = torch.tensor(
                np.array([d["policy_target"] for d in batch], dtype=np.float32),
                device=device
            )
            value_targets = torch.tensor(
                np.array([d["value_target"] for d in batch], dtype=np.float32),
                device=device
            )

            # Forward pass
            policy_pred, value_pred = network(positions)

            # Compute losses
            # CrossEntropy expects class indices, so use argmax of one-hot
            policy_targets_idx = policy_targets.argmax(dim=1)
            policy_loss = policy_loss_fn(policy_pred, policy_targets_idx)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            total_loss = 0.7 * policy_loss + 0.3 * value_loss

            #print(policy_pred.shape)   # Should be (B, 4096)
            #print(policy_targets_idx.shape)  # Should be (B,)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {i//batch_size+1}, "
                    f"Loss: {total_loss.item():.4f}"
                    f" (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})"
                )

        print(f"Epoch {epoch+1}/{epochs} completed")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(network.state_dict(), "models/trained_model.pth")
    print("Model saved to models/trained_model.pth")


# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    train()
