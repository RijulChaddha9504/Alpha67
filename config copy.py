# config.py - Configuration file for Chess Engine
# Optimized for better training performance

# ========================================
# Network Architecture
# ========================================
INPUT_CHANNELS = 13  # 6 piece types × 2 colors + en passant
HIDDEN_CHANNELS = 128  # Reduced from 256 for faster training
RESIDUAL_BLOCKS = 8  # Reduced from 10-12 for efficiency
POLICY_OUTPUT_SIZE = 4096  # 64×64 (all possible from-to square combinations)

# ========================================
# MCTS Configuration
# ========================================
NUM_SIMULATIONS = 100  # Balanced between strength and speed
C_PUCT = 1.5  # Exploration constant (higher = more exploration)

# ========================================
# Training Configuration
# ========================================
BATCH_SIZE = 128  # Batch size for training
LEARNING_RATE = 0.0003  # Learning rate (reduced for stability)
REPLAY_BUFFER_SIZE = 50000  # Maximum positions to keep in memory

# Loss weights
POLICY_LOSS_WEIGHT = 0.7
VALUE_LOSS_WEIGHT = 0.3

# ========================================
# Model Paths
# ========================================
MODEL_PATH = "models/trained_model.pth"
CHECKPOINT_DIR = "checkpoints"

# ========================================
# Training Schedule
# ========================================
# Default cycle configuration
DEFAULT_STOCKFISH_GAMES = 5
DEFAULT_SELF_PLAY_GAMES = 3
DEFAULT_TRAINING_BATCHES = 50

# Progressive difficulty
STOCKFISH_START_LEVEL = 1
STOCKFISH_MAX_LEVEL = 10
WIN_RATE_THRESHOLD = 0.6  # Increase difficulty at 60% win rate
CYCLES_BEFORE_LEVEL_UP = 5

# ========================================
# Performance Optimizations
# ========================================
USE_MIXED_PRECISION = False  # Set to True if you have a modern GPU
GRADIENT_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4

# ========================================
# Debugging
# ========================================
VERBOSE = True
SAVE_EVERY_N_CYCLES = 1  # Save checkpoint every N cycles
LOG_TRAINING_STATS = True