#!/usr/bin/env python3
# dummy_uci.py - minimal UCI engine with basic pondering
import sys
import threading
import time
import random
import chess

board = chess.Board()
stop_event = threading.Event()
ponder_event = threading.Event()
move_to_play = None

def send(s):
    """Send output to UCI controller."""
    sys.stdout.write(s + "\n")
    sys.stdout.flush()

def handle_uci():
    send("id name DummyPy")
    send("id author You")
    send("uciok")

def handle_isready():
    send("readyok")

def parse_position(parts):
    """Parse 'position' command and update the board."""
    global board
    if len(parts) >= 2 and parts[1] == "startpos":
        board = chess.Board()
        moves_index = 2
    elif len(parts) >= 2 and parts[1] == "fen":
        fen = " ".join(parts[2:8])
        board = chess.Board(fen)
        moves_index = 8
    else:
        return

    # Apply moves after position setup
    if len(parts) > moves_index and parts[moves_index] == "moves":
        for mv in parts[moves_index+1:]:
            try:
                board.push_uci(mv)
            except Exception:
                pass

def think_and_play():
    """Worker thread to pick a move while supporting stop/ponder."""
    global move_to_play
    legal = list(board.legal_moves)
    if not legal:
        move_to_play = "0000"
        return

    # Simulate some "thinking"
    start_time = time.time()
    while not stop_event.is_set() and (time.time() - start_time) < 0.1:
        # optionally, pick a random move each iteration to mimic deeper search
        move_to_play = random.choice(legal)
        # send dummy info line (like Stockfish)
        send(f"info depth 1 seldepth 1 nodes 1 score cp 0 time {int((time.time() - start_time)*1000)} pv {move_to_play.uci()}")
        time.sleep(0.02)

    if move_to_play is None:
        move_to_play = random.choice(legal)

def handle_go(parts):
    """Handle 'go' command."""
    global move_to_play
    stop_event.clear()
    ponder_event.clear()
    move_to_play = None

    # start worker thread
    t = threading.Thread(target=think_and_play)
    t.start()
    t.join()  # wait for "thinking" to finish

    send(f"bestmove {move_to_play}")

def handle_stop():
    """Stop thinking (called by GUI if needed)."""
    stop_event.set()

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        cmd = parts[0]

        if cmd == "uci":
            handle_uci()
        elif cmd == "isready":
            handle_isready()
        elif cmd == "ucinewgame":
            board.reset()
            stop_event.clear()
        elif cmd == "position":
            parse_position(parts)
        elif cmd == "go":
            handle_go(parts)
        elif cmd == "stop":
            handle_stop()
        elif cmd == "quit":
            break

if __name__ == "__main__":
    main()
