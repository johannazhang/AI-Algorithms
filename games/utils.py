###############################################################################
# This file contains helper functions and the heuristic functions
# for our AI agents to play the Mancala game.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import sys

###############################################################################
### DO NOT MODIFY THE CODE BELOW

### Global Constants ###
TOP = 0
BOTTOM = 1

### Errors ###
class InvalidMoveError(RuntimeError):
    pass

class AiTimeoutError(RuntimeError):
    pass

### Functions ###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_opponent(player):
    if player == BOTTOM:
        return TOP
    return BOTTOM

### DO NOT MODIFY THE CODE ABOVE
###############################################################################


def heuristic_basic(board, player):
    """
    Compute the heuristic value of the current board for the current player 
    based on the basic heuristic function.

    :param board: the current board.
    :param player: the current player.
    :return: an estimated utility of the current board for the current player.
    """
    mancalas = board.mancalas
    return mancalas[player] - mancalas[get_opponent(player)]


def heuristic_advanced(board, player): 
    """
    Compute the heuristic value of the current board for the current player
    based on the advanced heuristic function.

    :param board: the current board object.
    :param player: the current player.
    :return: an estimated heuristic value of the current board for the current player.
    """
    pockets = board.pockets
    mancalas = board.mancalas
    opponent = get_opponent(player)

    stone_difference = mancalas[player] - mancalas[opponent]
    player_pockets = 0
    possible_captures = 0
    for i in range(len(pockets[player])):
        if pockets[player][i] > 0:
            player_pockets += 1
        elif pockets[opponent][i] > 0:
            possible_captures += 1
    
    total_score = mancalas[player] + mancalas[opponent]
    total_stones = 8 * board.dimension
    
    if total_score < (0.3 * total_stones):
        return (0.5 * stone_difference) + (0.25 * player_pockets) + (0.25 * possible_captures)
    elif total_score < (0.7 * total_stones):
        return (0.25 * stone_difference) + (0.25 * player_pockets) + (0.5 * possible_captures)
    return (0.25 * stone_difference) + (0.5 * player_pockets) + (0.25 * possible_captures)