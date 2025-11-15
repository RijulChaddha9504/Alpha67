import torch
import chess
from network_architecture import ChessNetwork
from board_state import board_to_tensor

# Initialize network
network = ChessNetwork()
network.eval()  # evaluation mode

# Create a starting board
board = chess.Board()

# Convert board to tensor (channels-first: 13x8x8)
board_tensor = board_to_tensor(board)
board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).float()  # add batch dimension: 1x13x8x8

# Forward pass
with torch.no_grad():
    policy, value = network(board_tensor)

print("Policy output shape:", policy.shape)  # should be (1, POLICY_OUTPUT_SIZE)
print("Value output shape:", value.shape)    # should be (1, 1)
print("Sample policy probabilities (first 10):", policy[0][:10])
print("Sample value output:", value.item())
