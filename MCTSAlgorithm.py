import numpy as np
import torch
import chess
from board_state import board_to_tensor
from config import NUM_SIMULATIONS, C_PUCT
from network_architecture import network_output_to_move_probs


class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior_prob=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = {}  # move -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob  # From policy network
        self.repetition_count = 0

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        """Average value from all visits"""
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count


class MCTS:
    def __init__(self, network, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT,
                 device=None, eval_batch_size=16, use_autocast=True, debug=False):
        self.network = network
        self.num_simulations = int(num_simulations)
        self.c_puct = c_puct
        self.device = device if device else torch.device('cpu')
        self.eval_batch_size = int(eval_batch_size)
        self.use_autocast = use_autocast and (self.device.type == 'cuda')
        self.debug = debug  # <-- store debug flag
        # Ensure network stays on device once
        try:
            self.network.to(self.device)
        except Exception:
            pass

    def search(self, board):
        root = MCTSNode(board)

        '''if self.debug:
            print(f"[DEBUG] Starting MCTS search for {self.num_simulations} simulations")'''

        # expand root once (so root has priors)
        if not root.board.is_game_over():
            self._expand_node_single(root)
        self._add_exploration_noise(root)

        pending_leaves = []

        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            while not node.is_leaf() and not node.board.is_game_over():
                node = self._select_child(node)
                search_path.append(node)

            if node.board.is_game_over():
                val = self._result_to_value(node.board.result(), node.board.turn)
                self._backpropagate(search_path, val)
                '''if self.debug:
                    print(f"[DEBUG] Terminal node reached: result={node.board.result()}, value={val}")'''
                continue

            pending_leaves.append((node, search_path))

            if len(pending_leaves) >= self.eval_batch_size or sim == (self.num_simulations - 1):
                self._evaluate_and_expand_batch(pending_leaves)
                '''if self.debug:
                    print(f"[DEBUG] Evaluated batch of {len(pending_leaves)} leaves at simulation {sim+1}")'''
                pending_leaves = []

        if self.debug:
            '''print(f"[DEBUG] MCTS search completed. Root children visits:")
            for move, child in root.children.items():
                print(f"  Move: {move}, visits: {child.visit_count}, value: {child.value():.3f}")'''

        return (root, self._best_move(root))
    def _add_exploration_noise(self, node, alpha=0.3, frac=0.25):
        """Add Dirichlet noise to root node priors (for exploration)."""
        moves = list(node.children.keys())
        if not moves:
            return

        noise = np.random.dirichlet([alpha] * len(moves))
        for move, n in zip(moves, noise):
            old_prior = node.children[move].prior_prob
            node.children[move].prior_prob = (1 - frac) * old_prior + frac * n
            '''if self.debug:
                print(f"[DEBUG] Noise applied to move {move}: old={old_prior:.3f}, noise={n:.3f}, new={node.children[move].prior_prob:.3f}")'''


    def _evaluate_and_expand_batch(self, pending):
        """
        pending: list of (node, path)
        We will:
          - deduplicate identical node objects (same id)
          - build a batch of board tensors
          - forward the batch through the network once
          - expand children using the policy outputs and backpropagate using the values
        """
        if not pending:
            return

        # Deduplicate nodes (same node object may appear multiple times in pending)
        unique_nodes = {}
        node_paths = {}  # node_id -> list of paths that reached it
        ordered_nodes = []
        for node, path in pending:
            nid = id(node)
            if nid not in unique_nodes:
                unique_nodes[nid] = node
                node_paths[nid] = []
                ordered_nodes.append(nid)
            node_paths[nid].append(path)

        # Build batch tensors
        tensors = []
        for nid in ordered_nodes:
            node = unique_nodes[nid]
            bt = board_to_tensor(node.board)  # numpy
            tensors.append(torch.from_numpy(bt).unsqueeze(0).float())

        batch_tensor = torch.cat(tensors, dim=0).to(self.device)  # shape (B,C,H,W)

        # Run batched forward pass
        self.network.eval()
        with torch.no_grad():
            if self.use_autocast:
                from torch.cuda.amp import autocast
                with autocast():
                    policy_batch, value_batch = self.network(batch_tensor)
            else:
                policy_batch, value_batch = self.network(batch_tensor)

        # Ensure policy/value on CPU/numpy for network_output_to_move_probs (which expects numpy)
        policy_batch_np = policy_batch.detach().cpu().numpy()  # shape (B, policy_dim)
        value_batch_np = value_batch.detach().cpu().numpy().reshape(-1)  # shape (B,)

        # For each unique node, expand children and backpropagate for each path that reached it
        for i, nid in enumerate(ordered_nodes):
            node = unique_nodes[nid]
            policy_probs = policy_batch_np[i]
            value = float(value_batch_np[i])

            # Use user's helper to transform network policy output to {move:prob}
            move_probs = network_output_to_move_probs(node.board, policy_probs)

            # Create children nodes (only once)
            for move, prior in move_probs.items():
                # avoid re-adding an existing child (safety)
                if move in node.children:
                    continue
                child_board = node.board.copy()
                child_board.push(move)
                child_node = MCTSNode(board=child_board, parent=node, move=move, prior_prob=prior)
                node.children[move] = child_node

            # Backpropagate value for each path that arrived at this node
            for path in node_paths[nid]:
                self._backpropagate(path, value)

    def _select_child(self, node):
        if not node.children:
            return node

        best_score = -float('inf')
        best_child = None
        sqrt_parent = np.sqrt(node.visit_count) if node.visit_count > 0 else 1.0

        for move, child in node.children.items():
            Q = child.value()
            U = (self.c_puct * child.prior_prob * sqrt_parent / (1 + child.visit_count))
            repeat_penalty = 0
            if child.board.is_repetition(1):
                child.repetition_count += 1
                repeat_penalty = 0.5 * child.repetition_count

            score = Q + U - repeat_penalty

            '''if self.debug:
                print(f"[DEBUG] Move {move}: Q={Q:.3f}, U={U:.3f}, penalty={repeat_penalty}, score={score:.3f}")'''

            if score > best_score:
                best_score = score
                best_child = child

        return best_child



    def _result_to_value(self, result, turn):
        if result == "1-0":
            return 1 if turn else -1
        elif result == "0-1":
            return -1 if turn else 1
        else:
            return -0.3

    def _expand_node_single(self, node):
        """
        A single-node expansion helper (used initially for the root).
        """
        bt = board_to_tensor(node.board)
        bt_t = torch.from_numpy(bt).unsqueeze(0).float().to(self.device)
        self.network.eval()
        with torch.no_grad():
            policy_out, value_out = self.network(bt_t)
        policy_np = policy_out[0].cpu().numpy()
        value = float(value_out[0].item())
        move_probs = network_output_to_move_probs(node.board, policy_np)
        for move, prior in move_probs.items():
            child_board = node.board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(board=child_board, parent=node, move=move, prior_prob=prior)
        return value

    def _expand_node(self, node):
        """
        Backwards compatibility wrapper for older code calling `_expand_node`.
        Now it just calls the optimized single-node version.
        """
        return self._expand_node_single(node)


    def _backpropagate(self, search_path, value):
        """
        Update visit counts and values for all nodes in path
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip for parent's perspective

    def _best_move(self, root):
        """
        Select move with highest visit count
        """
        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        return best_move