import chess
import chess.engine
from chess_engine import ChessEngine

# --- Start the dummy engine as a UCI process ---
opponent = chess.engine.SimpleEngine.popen_uci(["python3", "simple_engine.py"])

# --- Initialize your neural network + MCTS engine ---
my_engine = ChessEngine()
print("✅ ChessEngine initialized.")

# --- Play a short match ---
board = chess.Board()
move_count = 0

while not board.is_game_over() and move_count < 400:
    print("\n", board, "\n")

    if board.turn == chess.WHITE:
        # Your engine plays as White
        move = my_engine.get_best_move(board)
        print("🧠 MyEngine plays:", move)
    else:
        # The dummy engine plays as Black
        result = opponent.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        print("🤖 DummyEngine plays:", move)

    if move is None:
        break

    board.push(move)
    move_count += 1

print("\n🏁 Game Over! Result:", board.result())

opponent.quit()

