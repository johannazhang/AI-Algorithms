############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
##
############################################################


def prop_FC(csp, last_assigned_var=None):
    """
    This is a propagator to perform forward checking. 

    First, collect all the relevant constraints.
    If the last assigned variable is None, then no variable has been assigned 
    and we are performing propagation before search starts.
    In this case, we will check all the constraints.
    Otherwise, we will only check constraints involving the last assigned variable.

    Among all the relevant constraints, focus on the constraints with one unassigned variable. 
    Consider every value in the unassigned variable's domain, if the value violates 
    any constraint, prune the value. 

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: The boolean indicates whether forward checking is successful.
        The boolean is False if at least one domain becomes empty after forward checking.
        The boolean is True otherwise.
        Also returns a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """
    if last_assigned_var:
        constraints = csp.get_cons_with_var(last_assigned_var)
    else:
        constraints = csp.get_all_cons()

    pruned = []
    for constraint in constraints:
        if constraint.get_num_unassigned_vars() != 1:
            continue
 
        variable = constraint.get_unassigned_vars()[0]
        cur_domain = variable.cur_domain()
        for val in cur_domain:
            scope = constraint.get_scope()
            if scope[0] == variable:
                check_vals = [val, scope[1].get_assigned_value()]
            else:
                check_vals = [scope[0].get_assigned_value(), val]

            if not constraint.check(check_vals):
                variable.prune_value(val)
                pruned.append((variable, val))
                if len(cur_domain) == 0:
                    return False, pruned
                
    return True, pruned


def prop_AC3(csp, last_assigned_var=None):
    """
    This is a propagator to perform the AC-3 algorithm.

    Keep track of all the constraints in a queue (list). 
    If the last_assigned_var is not None, then we only need to 
    consider constraints that involve the last assigned variable.

    For each constraint, consider every variable in the constraint and 
    every value in the variable's domain.
    For each variable and value pair, prune it if it is not part of 
    a satisfying assignment for the constraint. 
    Finally, if we have pruned any value for a variable,
    add other constraints involving the variable back into the queue.

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes 
        all the constraints and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """
    
    if last_assigned_var:
        constraints = csp.get_cons_with_var(last_assigned_var)
    else:
        constraints = csp.get_all_cons()

    pruned = []
    while len(constraints) != 0:
        constraint = constraints.pop(0)
        for variable in constraint.get_scope():
            for val in variable.cur_domain():
                if not check_satisfying_tuple(constraint, variable, val):
                    variable.prune_value(val)
                    pruned.append((variable, val))
                    constraints.extend(csp.get_cons_with_var(variable))      
                    if variable.cur_domain_size() == 0:
                        return False, pruned
        
    return True, pruned


def check_satisfying_tuple(constraint, variable, value):
    """
    Check if the given value is part of any satisfying tuple for the constraint,
    using the current domain of the variable.
    """

    sup_tuples = constraint.sup_tuples.get((variable, value), [])
    for sup_tuple in sup_tuples:
        all_in_domain = True
        for i in range(len(sup_tuple)):
            if not constraint.scope[i].in_cur_domain(sup_tuple[i]):
                all_in_domain = False
                break
        if all_in_domain:
            return True
    return False


def ord_mrv(csp):
    """
    Implement the Minimum Remaining Values (MRV) heuristic.
    Choose the next variable to assign based on MRV.

    If there is a tie, we will choose the first variable. 

    :param csp: A CSP problem
    :type csp: CSP

    :returns: the next variable to assign based on MRV

    """

    variables = csp.get_all_unasgn_vars()

    mrv = None
    min_domain = float("inf")
    for variable in variables:
        domain_size = variable.domain_size()
        if domain_size < min_domain:
            min_domain = domain_size
            mrv = variable
    
    return mrv


###############################################################################
# Do not modify the prop_BT function below
###############################################################################


def prop_BT(csp, last_assigned_var=None):
    """
    This is a basic propagator for plain backtracking search.

    Check if the current assignment satisfies all the constraints.
    Note that we only need to check all the fully instantiated constraints 
    that contain the last assigned variable.
    
    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes all the constraints 
        and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]

    """
    
    # If we haven't assigned any variable yet, return true.
    if not last_assigned_var:
        return True, []
        
    # Check all the constraints that contain the last assigned variable.
    for c in csp.get_cons_with_var(last_assigned_var):

        # All the variables in the constraint have been assigned.
        if c.get_num_unassigned_vars() == 0:

            # get the variables
            vars = c.get_scope() 

            # get the list of values
            vals = []
            for var in vars: #
                vals.append(var.get_assigned_value())

            # check if the constraint is satisfied
            if not c.check(vals): 
                return False, []

    return True, []
