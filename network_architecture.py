import torch
import torch.nn as nn
from config import INPUT_CHANNELS, RESIDUAL_BLOCKS, HIDDEN_CHANNELS, POLICY_OUTPUT_SIZE
import numpy as np

class ChessNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared backbone
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(HIDDEN_CHANNELS)
        self.conv3 = nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(HIDDEN_CHANNELS)
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_CHANNELS) for _ in range(RESIDUAL_BLOCKS)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_OUTPUT_SIZE)

        
        # Value head
        self.value_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_fc1_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        # Shared processing
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)
        # NOTE: No softmax here — use raw logits for CrossEntropyLoss

        # Value head
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1_bn(self.value_fc1(value)))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out


def network_output_to_move_probs(board, policy_output):
    """
    Convert raw logits into a normalized probability distribution
    over legal moves.
    """

    # --- Convert input to numpy safely ---
    if isinstance(policy_output, torch.Tensor):
        logits = policy_output.detach().cpu().numpy()
    else:
        logits = np.array(policy_output)

    # --- Softmax over all logits ---
    exp_logits = np.exp(logits - np.max(logits))  # numerically stable
    softmax_probs = exp_logits / np.sum(exp_logits)

    legal_moves = list(board.legal_moves)

    # Map move → index in policy vector
    move_to_idx = {m: (m.from_square * 64 + m.to_square) for m in legal_moves}

    # Extract only legal move probabilities
    legal_move_probs = {m: softmax_probs[idx] for m, idx in move_to_idx.items()}

    # Normalize so legal move probabilities sum to 1
    total = sum(legal_move_probs.values())

    if total > 0:
        legal_move_probs = {m: p / total for m, p in legal_move_probs.items()}
    else:
        # Fallback uniform distribution (should never happen but safe)
        uniform = 1 / len(legal_moves)
        legal_move_probs = {m: uniform for m in legal_moves}

    return legal_move_probs