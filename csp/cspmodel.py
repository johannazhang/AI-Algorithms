############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.1
## Changes:
##   v1.1: updated the comments in kropki_model. 
##         the second return value should be a 2d list of variables.
############################################################

from board import *
from cspbase import *

def kropki_model(board):
    """
    Create a CSP for a Kropki Sudoku Puzzle given a board of dimension.

    If a variable has an initial value, its domain should only contain the initial value.
    Otherwise, the variable's domain should contain all possible values (1 to dimension).

    We will encode all the constraints as binary constraints.
    Each constraint is represented by a list of tuples, representing the values that
    satisfy this constraint. (This is the table representation taught in lecture.)

    Remember that a Kropki sudoku has the following constraints.
    - Row constraint: every two cells in a row must have different values.
    - Column constraint: every two cells in a column must have different values.
    - Cage constraint: every two cells in a 2x3 cage (for 6x6 puzzle) 
            or 3x3 cage (for 9x9 puzzle) must have different values.
    - Black dot constraints: one value is twice the other value.
    - White dot constraints: the two values are consecutive (differ by 1).

    Make sure that you return a 2D list of variables separately. 
    Once the CSP is solved, we will use this list of variables to populate the solved board.
    Take a look at csprun.py for the expected format of this 2D list.

    :returns: A CSP object and a list of variables.
    :rtype: CSP, List[List[Variable]]

    """
    dim = board.dimension
    domain = create_initial_domain(dim)

    variables = []
    for i in range(dim):
        for j in range(dim):
            if board.cells[i][j] != 0:
                variable = Variable(f"Var({i}, {j})", [board.cells[i][j]])
            else:
                variable = Variable(f"Var({i}, {j})", domain)
            variables.append(variable)

    variables_list = []
    for i in range(dim):
        variables_list.append(variables[i*dim:(i*dim)+dim])

    row_and_col_constraints = create_row_and_col_constraints(dim, satisfying_tuples_difference_constraints(dim), variables)
    cage_constraints = create_cage_constraints(dim, satisfying_tuples_difference_constraints(dim), variables)
    dot_constraints = create_dot_constraints(dim, board.dots, satisfying_tuples_white_dots(dim), satisfying_tuples_black_dots(dim), variables)
    
    csp = CSP("kropki", variables)
    for constraint in row_and_col_constraints:
        csp.add_constraint(constraint)
    for constraint in cage_constraints:
        csp.add_constraint(constraint)
    for constraint in dot_constraints:
        csp.add_constraint(constraint)

    return csp, variables_list
    
    
    
def create_initial_domain(dim):
    """
    Return a list of values for the initial domain of any unassigned variable.
    [1, 2, ..., dimension]

    :param dim: board dimension
    :type dim: int

    :returns: A list of values for the initial domain of any unassigned variable.
    :rtype: List[int]
    """

    domain = []
    for i in range(dim):
        domain.append(i+1)
    return domain



def create_variables(dim):
    """
    Return a list of variables for the board.

    We recommend that your name each variable Var(row, col).

    :param dim: Size of the board
    :type dim: int

    :returns: A list of variables, one for each cell on the board
    :rtype: List[Variables]
    """

    domain = create_initial_domain(dim)
    variables = []
    for i in range(dim):
        for j in range(dim):
            variable = Variable(f"Var({i}, {j})", domain)
            variables.append(variable)
    return variables

    
def satisfying_tuples_difference_constraints(dim):
    """
    Return a list of satifying tuples for binary difference constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    tuples = []
    for i in range(1, dim+1):
        for j in range(1, dim+1):
            if i != j:
                tuples.append((i, j))
    return tuples
  
  
def satisfying_tuples_white_dots(dim):
    """
    Return a list of satifying tuples for white dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    tuples = []
    for i in range(1, dim+1):
        for j in range(1, dim+1):
            if abs(i - j) == 1:
                tuples.append((i, j))
    return tuples

  
def satisfying_tuples_black_dots(dim):
    """
    Return a list of satifying tuples for black dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    tuples = []
    for i in range(1, dim+1):
        for j in range(1, dim+1):
            if i*2 == j or j*2 == i:
                tuples.append((i, j))
    return tuples

    
def create_row_and_col_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different row/column constraints.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """
    
    constraints = []
    for row in range(dim):
        for col1 in range (dim):
          for col2 in range (col1+1, dim):
              row_constraint = Constraint(f"C(Var({row}, {col1}), Var({row}, {col2}))", [variables[row*dim + col1], variables[row*dim + col2]])
              row_constraint.add_satisfying_tuples(sat_tuples)
              constraints.append(row_constraint)
              col_constraint = Constraint(f"C(Var({col1}, {row}), Var({col2}, {row}))", [variables[col1*dim + row], variables[col2*dim + row]])
              col_constraint.add_satisfying_tuples(sat_tuples)
              constraints.append(col_constraint) 

    return constraints
    
    
def create_cage_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different constraints for all cages.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    constraints = []
    if dim == 6:
        for x in range(2):
            for y in range(3):

                cage_cells = []
                for row in range(3):
                    for col in range(2):
                        cage_cells.append((row + x*3, col + y*2))

                for i in range(dim):
                    for j in range(i+1, dim):
                        row1, col1 = cage_cells[i][0], cage_cells[i][1],
                        row2, col2 = cage_cells[j][0], cage_cells[j][1],
                        constraint = Constraint(f"C(Var({row1}, {col1}), Var({row2}, {col2}))", [variables[row1*dim + col1], variables[row2*dim + col2]])
                        constraint.add_satisfying_tuples(sat_tuples)
                        constraints.append(constraint)
    elif dim == 9:
        for x in range(3):
            for y in range(3):

                cage_cells = []
                for row in range(3):
                    for col in range(3):
                        cage_cells.append((row + x*3, col + y*3))

                for i in range(dim):
                    for j in range(i+1, dim):
                        row1, col1 = cage_cells[i][0], cage_cells[i][1],
                        row2, col2 = cage_cells[j][0], cage_cells[j][1],
                        constraint = Constraint(f"C(Var({row1}, {col1}), Var({row2}, {col2}))", [variables[row1*dim + col1], variables[row2*dim + col2]])
                        constraint.add_satisfying_tuples(sat_tuples)
                        constraints.append(constraint)

    return constraints

    
def create_dot_constraints(dim, dots, white_tuples, black_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]

    :param white_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the white dot constraint.
    :type white_tuples: List[(int, int)]
    
    :param black_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the black dot constraint.
    :type black_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary dot constraints
    :rtype: List[Constraint]
    """

    constraints = []
    for dot in dots:
        row1, col1 = dot.cell_row, dot.cell_col 
        row2, col2 = dot.cell2_row, dot.cell2_col
        constraint = Constraint(f"C(Var({row1}, {col1}), Var({row2}, {col2}))", [variables[row1*dim + col1], variables[row2*dim + col2]])
        if dot.color == CHAR_BLACK:
            constraint.add_satisfying_tuples(black_tuples)
        else:
            constraint.add_satisfying_tuples(white_tuples)
        constraints.append(constraint)

    return constraints

