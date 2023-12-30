import copy
import math
import random
from math import log, sqrt, inf
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress

import classes.logic as logic

# Base class for different player strategies in the game.
class PlayerStrat:
    def __init__(self, _board_state, player):
        """
        Initialize the player strategy with the current state of the board and the player number.

        :param _board_state: The current state of the board as a 2D list.
        :param player: The player number (1 or 2).
        """
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        Abstract method to select a tile from the board. To be implemented by subclasses.

        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        raise NotImplementedError

# Random strategy for a player. Chooses a move randomly from available tiles.
class RandomPlayer(PlayerStrat):
    def __init__(self, _board_state, player):
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)

    def select_tile(self, board):
        """
        Randomly selects a free tile on the board.

        :param board: The current game board.
        :returns: (x, y) tuple of integers corresponding to a valid and free tile on the board.
        """
        free_tiles = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if board[x][y] == 0]
        return random.choice(free_tiles) if free_tiles else None

    def start(self):
        return self.select_tile(self.root_state)

# MiniMax strategy for a player. Uses the MiniMax algorithm to choose the best move.
class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, depth=5):
        """
        Initialize the MiniMax player with the current state of the board, player number, and search depth.

        :param _board_state: The current state of the board as a 2D list.
        :param player: The player number (1 or 2).
        :param depth: The depth of search in the MiniMax algorithm.
        """
        super().__init__(_board_state, player)
        self.board_size = len(_board_state)
        self.depth = depth

    def select_tile(self, board, player):
        """
        Selects the best move using the MiniMax strategy.

        :param board: The current game board.
        :param player: The player number for whom the move is being calculated.
        :returns: (x, y) tuple of integers corresponding to the best move.
        """
        best_score = float('-inf')
        best_move = None

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x][y] == 0:
                    board[x][y] = player
                    score = self.minimax(board, self.depth - 1, False, player)
                    board[x][y] = 0
                    if score > best_score:
                        best_score = score
                        best_move = (x, y)

        return best_move

    def minimax(self, board, depth, is_maximizing, player, alpha=float('-inf'),      beta=float('inf')):
        """
        MiniMax algorithm with alpha-beta pruning to evaluate the board and return the best score.

        :param board: The game board.
        :param depth: Current depth of the search tree.
        :param is_maximizing: Boolean indicating whether the current layer is maximizing or minimizing.
        :param player: The player number.
        :param alpha: The best value that the maximizer currently can guarantee at that level or above.
        :param beta: The best value that the minimizer currently can guarantee at that level or above.
        :returns: The best score from the current board state.
        """
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board, player)

        if is_maximizing:
            best_score = float('-inf')
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = player
                        score = self.minimax(board, depth - 1, False, player, alpha, beta)
                        board[x][y] = 0
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
            return best_score
        else:
            best_score = float('inf')
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if board[x][y] == 0:
                        board[x][y] = 3 - player
                        score = self.minimax(board, depth - 1, True, player, alpha, beta)
                        board[x][y] = 0
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
            return best_score

    def is_game_over(self, board):
        """
        Check if the game is over.

        :param board: The game board.
        :returns: Boolean indicating whether the game is over.
        """
        return logic.is_game_over(self.player, board) is not None

    def evaluate_board(self, board, player):
        """
        Evaluate the board and return a score based on the game state.

        :param board: The game board.
        :param player: The player number.
        :returns: A score representing the state of the board.
        """
        if logic.is_game_over(player, board):
            return 10  # Win
        elif logic.is_game_over(3 - player, board):
            return -10  # Loss
        else:
            return 0  # Neutral

    def start(self):
        return self.select_tile(self.root_state, self.player)

# Dictionary to map strategy names to their respective classes.
str2strat: dict[str, PlayerStrat] = {
    "human": None,
    "random": RandomPlayer,
    "minimax": MiniMax,
}
