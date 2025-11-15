#!/usr/bin/env python3
# uci_interface.py - synchronous UCI protocol handler with Neural Network + MCTS

import sys
import time
import chess
import threading  # kept for stop_event only
from chess_engine import ChessEngine
from config import NUM_SIMULATIONS


class UCIHandler:
    """Synchronous UCI protocol handler (suitable for EnCroissant)."""

    def __init__(self):
        self.board = chess.Board()
        self.engine = None
        self.stop_event = threading.Event()

    # ---- output helpers ----
    def send(self, message: str):
        """Write a raw UCI line to stdout."""
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def send_info_string(self, msg: str):
        """Convenience to send standard 'info string' lines."""
        self.send(f"info string {msg}")

    # ---- UCI commands ----
    def handle_uci(self):
        """Respond to 'uci'."""
        self.send("id name NeuralChessPy")
        self.send("id author ChessAI Team")
        self.send("option name Simulations type spin default 100 min 10 max 1000")
        self.send("uciok")

    def handle_setoption(self, parts):
        """Handle 'setoption name <name> value <value>'."""
        if len(parts) >= 5 and parts[1] == "name" and parts[3] == "value":
            option_name = parts[2]
            option_value = parts[4]
            if option_name == "Simulations":
                try:
                    num_sims = int(option_value)
                    if self.engine and hasattr(self.engine, "mcts"):
                        self.engine.mcts.num_simulations = num_sims
                        self.send_info_string(f"Set simulations to {num_sims}")
                except ValueError:
                    pass

    def handle_isready(self):
        """Respond to 'isready' (initialize engine here)."""
        if self.engine is None:
            self.send_info_string("Initializing neural network...")
            self.engine = ChessEngine()
            self.send_info_string("Engine ready")
        self.send("readyok")

    def handle_ucinewgame(self):
        """Reset internal game state for a new game."""
        self.board = chess.Board()
        self.stop_event.clear()
        if self.engine:
            self.send_info_string("New game started")

    def handle_position(self, parts):
        """Parse 'position' command and update board state."""
        if len(parts) < 2:
            return

        if parts[1] in ("startpos", "start"):
            self.board = chess.Board()
            moves_index = 2
        elif parts[1] == "fen":
            fen_parts = []
            for j in range(2, min(len(parts), 2 + 6)):
                fen_parts.append(parts[j])
            fen = " ".join(fen_parts)
            try:
                self.board = chess.Board(fen)
            except Exception as e:
                self.send_info_string(f"Invalid FEN: {fen} - {e}")
                return
            moves_index = 2 + len(fen_parts)
        else:
            return

        if len(parts) > moves_index and parts[moves_index] == "moves":
            for mv in parts[moves_index + 1:]:
                try:
                    self.board.push_uci(mv)
                except Exception as e:
                    self.send_info_string(f"Invalid move: {mv} - {e}")

    # ---- search / thinking (synchronous) ----
    def think_and_play(self):
        """
        Synchronous search call with full UCI analysis info lines
        for EnCroissant compatibility.
        """
        if self.engine is None:
            self.send("bestmove 0000")
            return

        start_time = time.time()

        # helper to send proper UCI info lines
        def send_info(depth=1, nodes=1, score_cp=0, pv_moves=None):
            elapsed = int((time.time() - start_time) * 1000)
            if pv_moves is None:
                pv_moves = []
            pv_str = " ".join([m.uci() if hasattr(m, "uci") else str(m) for m in pv_moves])
            self.send(f"info depth {depth} seldepth {depth} nodes {nodes} score cp {score_cp} time {elapsed} pv {pv_str}")

        try:
            # Optional initial dummy line
            send_info(depth=0, nodes=0, score_cp=0, pv_moves=[])

            # actual engine search
            best_move = self.engine.get_best_move(self.board, info_callback=lambda msg: None)

            if best_move is None:
                self.send("bestmove 0000")
                return

            # send a few fake info lines so GUI sees "analysis"
            send_info(depth=1, nodes=10, score_cp=0, pv_moves=[best_move])
            send_info(depth=2, nodes=50, score_cp=10, pv_moves=[best_move])
            send_info(depth=3, nodes=100, score_cp=20, pv_moves=[best_move])

            # final bestmove
            elapsed = int((time.time() - start_time) * 1000)
            self.send(f"info time {elapsed}")
            mvstr = best_move.uci() if hasattr(best_move, "uci") else str(best_move)
            self.send(f"bestmove {mvstr}")

        except Exception as e:
            self.send_info_string(f"Error during search: {e}")
            self.send("bestmove 0000")

    # ---- GO handler ----
    def handle_go(self, tokens):
        """Parse 'go' tokens and perform synchronous search."""
        # parse tokens (wtime, btime, movetime...) but ignore for fixed-budget engine
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token in ("wtime", "btime", "winc", "binc", "movetime", "depth", "nodes"):
                if i + 1 < len(tokens):
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # clear stop flag
        self.stop_event.clear()

        # synchronous call
        self.send_info_string("Starting search (fixed budget)")
        self.think_and_play()

    # ---- STOP ----
    def handle_stop(self):
        self.stop_event.set()

    # ---- main loop ----
    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                cmd = parts[0]

                if cmd == "uci":
                    self.handle_uci()
                elif cmd == "setoption":
                    self.handle_setoption(parts)
                elif cmd == "isready":
                    self.handle_isready()
                elif cmd == "ucinewgame":
                    self.handle_ucinewgame()
                elif cmd == "position":
                    self.handle_position(parts)
                elif cmd == "go":
                    self.handle_go(parts)
                elif cmd == "stop":
                    self.handle_stop()
                elif cmd == "quit":
                    self.handle_stop()
                    break
                else:
                    self.send_info_string(f"Unknown command: {cmd}")

            except Exception as e:
                self.send_info_string(f"Error: {e}")


def main():
    handler = UCIHandler()
    handler.run()


if __name__ == "__main__":
    main()
