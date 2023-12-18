###############################################################################
# This file implements various minimax search agents.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################
from mancala_game import Board, play_move
from utils import *


def minimax_max_basic(board, curr_player, heuristic_func):
    """
    Perform Minimax Search for MAX player.
    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :return the best move and its minimax value according to minimax search.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent): 
        return None, heuristic_func(board, curr_player)
    
    best_move, best_value = None, float('-inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_min_basic(next_state, opponent, heuristic_func)[1]
        if value > best_value:
            best_move, best_value = move, value

    return best_move, best_value


def minimax_min_basic(board, curr_player, heuristic_func):
    """
    Perform Minimax Search for MIN player.
    Return the best move and its minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :return the best move and its minimax value according to minimax search.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent): 
        return None, heuristic_func(board, opponent)
    
    best_move, best_value = None, float('inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_max_basic(next_state, opponent, heuristic_func)[1]
        if value < best_value:
            best_move, best_value = move, value

    return best_move, best_value


def minimax_max_limit(board, curr_player, heuristic_func, depth_limit):
    """
    Perform Minimax Search for MAX player up to the given depth limit.
    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its minimmax value estimated by our heuristic function.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent) or depth_limit == 0: 
        return None, heuristic_func(board, curr_player)
    
    limit = depth_limit - 1
    best_move, best_value = None, float('-inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_min_limit(next_state, opponent, heuristic_func, limit)[1]
        if value > best_value:
            best_move, best_value = move, value

    return best_move, best_value


def minimax_min_limit(board, curr_player, heuristic_func, depth_limit):
    """
    Perform Minimax Search for MIN player  up to the given depth limit.
    Return the best move and its estimated minimax value.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its minimmax value estimated by our heuristic function.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent) or depth_limit == 0: 
        return None, heuristic_func(board, opponent)
    
    limit = depth_limit - 1
    best_move, best_value = None, float('inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_max_limit(next_state, opponent, heuristic_func, limit)[1]
        if value < best_value:
            best_move, best_value = move, value

    return best_move, best_value


def minimax_max_limit_caching(board, curr_player, heuristic_func, depth_limit, cache):
    """
    Perform Minimax Search for MAX player up to the given depth limit with the option of caching states.
    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the ccurrent player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param caching: whether we are caching states.
    :return the best move and its minimmax value estimated by our heuristic function.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent) or depth_limit == 0: 
        return None, heuristic_func(board, curr_player)

    limit = depth_limit - 1
    if (board, curr_player) in cache:
        move, value, cached_limit = cache[(board, curr_player)]
        if cached_limit >= limit:
            return move, value

    best_move, best_value = None, float('-inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_min_limit_caching(next_state, opponent, heuristic_func, limit, cache)[1]
        if value > best_value:
            best_move, best_value = move, value

    cache[(board, curr_player)] = best_move, best_value, limit
    return best_move, best_value


def minimax_min_limit_caching(board, curr_player, heuristic_func, depth_limit, cache):
    """
    Perform Minimax Search for MIN player up to the given depth limit with the option of caching states.
    Return the best move and its minimmax value estimated by our heuristic function.
    If the board is a terminal state, return None as its best move.

    :param board: the current board
    :param curr_player: the current player
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param caching: whether we are caching states.
    :return the best move and its minimmax value estimated by our heuristic function.
    """
    opponent = get_opponent(curr_player)
    moves = board.get_possible_moves(curr_player)
    if not moves or not board.get_possible_moves(opponent) or depth_limit == 0: 
        return None, heuristic_func(board, opponent) 
    
    limit = depth_limit - 1
    if (board, curr_player) in cache:
        move, value, cached_limit = cache[(board, curr_player)]
        if cached_limit >= limit:
            return move, value

    best_move, best_value = None, float('inf')
    for move in moves:
        next_state = play_move(board, curr_player, move)
        value = minimax_max_limit_caching(next_state, opponent, heuristic_func, limit, cache)[1]
        if value < best_value:
            best_move, best_value = move, value

    cache[(board, curr_player)] = best_move, best_value, limit
    return best_move, best_value


###############################################################################
## DO NOT MODIFY THE CODE BELOW.
###############################################################################

def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Mancala AI")  # First line is the name of this AI
    arguments = input().split(",")

    player = int(arguments[0])  # Player color
    limit = int(arguments[1])  # Depth limit
    caching = int(arguments[2])  # Caching
    hfunc = int(arguments[3]) # Heuristic Function

    if (caching == 1): 
        caching = True
    else: 
        caching = False

    eprint("Running MINIMAX")


    if limit == -1:
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is", limit)

    if caching:
        eprint("Caching is ON")
        cache = {}
    else:
        eprint("Caching is OFF")

    if hfunc == 0:
        eprint("Using heuristic_basic")
        heuristic_func = heuristic_basic
    else:
        eprint("Using heuristic_advanced")
        heuristic_func = heuristic_advanced

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":  # Game is over.
            print()
        else:
            pockets = eval(input())  # Read in the input and turn it into an object
            mancalas = eval(input())  # Read in the input and turn it into an object
            board = Board(pockets, mancalas)

            # Select the move and send it to the manager
            if caching:
                move, value = minimax_max_limit_caching(board, player, heuristic_func, limit, cache)
            elif limit >= 0:
                move, value = minimax_max_limit(board, player, heuristic_func, limit)
            else:
                move, value = minimax_max_basic(board, player, heuristic_func)
            print("{}".format(move))


if __name__ == "__main__":
    run_ai()
