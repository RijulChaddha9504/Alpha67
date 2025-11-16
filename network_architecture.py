import torch
import torch.nn as nn
import numpy as np
from config import INPUT_CHANNELS, RESIDUAL_BLOCKS, HIDDEN_CHANNELS, POLICY_OUTPUT_SIZE


class ChessNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --------------------------
        # Shared backbone
        # --------------------------
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(HIDDEN_CHANNELS)

        self.conv3 = nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(HIDDEN_CHANNELS)

        # --------------------------
        # ResNet blocks
        # --------------------------
        self.res_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_CHANNELS) for _ in range(RESIDUAL_BLOCKS)
        ])

        # --------------------------
        # Policy head
        # --------------------------
        self.policy_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_OUTPUT_SIZE)

        # --------------------------
        # Value head
        # --------------------------
        self.value_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc1_bn = nn.BatchNorm1d(256)
        self.value_fc2 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.25)

        # Runtime flag
        self.optimized = False
    

    def forward(self, x):
        # Shared trunk
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        for block in self.res_blocks:
            x = block(x)

        # ----------------- Policy -----------------
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)  # logits (not softmaxed)

        # ----------------- Value ------------------
        value = torch.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1_bn(self.value_fc1(value)))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))  # range [-1,1]

        return policy, value


    # =====================
    # 🔥 Optimization Mode
    # =====================
    def enable_fast_inference(self):
        """
        Optimizes the model for repeated inference during MCTS:
        - Disables dropout
        - Freezes BatchNorm statistics
        - Compiles the network (if supported)
        """
        if self.optimized:
            return self
        
        # Freeze dropout + BN stats
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.eval()
                layer.track_running_stats = False
            if isinstance(layer, nn.Dropout):
                layer.p = 0.0  # disable dropout entirely

        # Try torch_compile (PyTorch 2.0+). Safe fallback.
        try:
            self = torch.compile(self)
        except Exception:
            pass

        self.optimized = True
        return self



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


# ==========================
# Legal move post-processing
# ==========================
def network_output_to_move_probs(board, policy_output):
    """
    Convert logits → softmax → legal move probability dictionary.
    """

    # Ensure policy is numpy
    if isinstance(policy_output, torch.Tensor):
        logits = policy_output.detach().cpu().numpy()
    else:
        logits = np.array(policy_output)

    # Softmax over full move space (4096 indexes)
    exp_logits = np.exp(logits - np.max(logits))
    softmax_probs = exp_logits / np.sum(exp_logits)

    legal_moves = list(board.legal_moves)
    move_to_idx = {m: m.from_square * 64 + m.to_square for m in legal_moves}

    legal_move_probs = {m: softmax_probs[idx] for m, idx in move_to_idx.items()}

    # Renormalize over legal moves
    S = sum(legal_move_probs.values())
    if S > 0:
        legal_move_probs = {m: p / S for m, p in legal_move_probs.items()}
    else:
        # Rare fallback: uniform random
        legal_move_probs = {m: 1 / len(legal_moves) for m in legal_moves}

    return legal_move_probs
