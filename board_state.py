import numpy as np
import chess

def board_to_tensor(board):
    """
    Convert chess.Board to neural network input from current player's perspective.
    Output shape: (13, 8, 8) -> channels-first for PyTorch
    
    🔥 CRITICAL: Board is always encoded from the perspective of the player to move.
    If Black is to move, the board is flipped so Black's pieces appear at the "bottom"
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Determine if we need to flip the board (Black's perspective)
    flip = not board.turn  # Flip if Black to move
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            
            # 🔥 FIX: Flip coordinates if viewing from Black's perspective
            if flip:
                rank = 7 - rank
                file = 7 - file
            
            plane_idx = piece_to_plane[piece.piece_type]
            
            # Current player's pieces in planes 0-5, opponent's in 6-11
            if piece.color == board.turn:
                tensor[plane_idx, rank, file] = 1.0
            else:
                tensor[plane_idx + 6, rank, file] = 1.0
    
    # Plane 12: metadata (castling rights for current player)
    if board.has_kingside_castling_rights(board.turn):
        tensor[12, :, :] = 0.1
    if board.has_queenside_castling_rights(board.turn):
        tensor[12, :, :] += 0.1
    if board.has_kingside_castling_rights(not board.turn):
        tensor[12, :, :] += 0.2
    if board.has_queenside_castling_rights(not board.turn):
        tensor[12, :, :] += 0.2
    
    return tensor