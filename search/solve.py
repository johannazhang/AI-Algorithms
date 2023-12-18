############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes:
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity
from collections import deque

from board import *

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    board = state.board
    for box in board.boxes:
        if box not in  board.storage:
            return False
    return True


def get_path(state):
    """
    Return a list of states containing the nodes on the path
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    path = [state]
    parent = state.parent
    while parent is not None:
        path.insert(0, parent)
        parent = parent.parent

    return path


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    successors = []
    board = state.board
    robots = board.robots
    storage = board.storage
    boxes = board.boxes
    obstacles = board.obstacles

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for robot in robots:
        for direction in directions:
            position = (robot[0] + direction[0], robot[1] + direction[1])
            adjacent = (robot[0] + 2*direction[0], robot[1] + 2*direction[1])

            if position in obstacles or position in robots: continue
            if position in boxes:
                if adjacent in obstacles or adjacent in robots or adjacent in boxes: continue
                new_boxes = [adjacent if b==position else b for b in boxes]
            else:
                new_boxes = boxes
            new_robots = [position if r==robot else r for r in robots]

            new_board = Board(board.name, board.width, board.height, new_robots, new_boxes, storage, obstacles)
            f = state.depth + 1 + state.hfn(new_board)
            new_state = State(new_board, state.hfn, f, state.depth + 1, state)
            successors.append(new_state)

    return successors


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    state = State(init_board, heuristic_zero, 0, 0, None)
    frontier = deque([state])
    explored = set()
    while len(frontier) > 0:
        state = frontier.pop()
        board = state.board

        if board in explored: continue
        explored.add(board)

        if is_goal(state):
            return get_path(state), state.depth
        successors = get_successors(state)
        frontier.extend(successors)

    return [], -1


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    state = State(init_board, hfn, hfn(init_board), 0, None)
    frontier = [state]
    heapq.heapify(frontier)
    explored = set()
    while len(frontier) > 0:
        state = heapq.heappop(frontier)
        board = state.board

        if board in explored: continue
        explored.add(board)

        if is_goal(state):
            return get_path(state), state.depth

        successors = get_successors(state)
        for s in successors:
            heapq.heappush(frontier, s)

    return [], -1


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    storage = board.storage
    boxes = board.boxes
    sum = 0
    for box in boxes:
        min_distance = float('inf')
        for store in storage:
            distance = abs(store[0]-box[0]) + abs(store[1]-box[1])
            if distance < min_distance:
                min_distance = distance
        sum += min_distance

    return sum


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    distance = heuristic_basic(board)
    storage = board.storage
    boxes = board.boxes
    obstacles = board.obstacles
    for box in boxes:
        if box in storage: continue
        x = box[0]
        y = box[1]
        # if box in corner
        up, right, down, left = (x-1, y), (x, y+1), (x+1, y), (x, y-1)
        if (up in obstacles and right in obstacles) \
            or (right in obstacles and down in obstacles) \
            or (down in obstacles and left in obstacles) \
            or (left in obstacles and up in obstacles):
            distance *= 1000

    return distance


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()
