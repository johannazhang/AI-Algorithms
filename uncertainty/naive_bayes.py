############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.2
## - removed the example in ve since it is misleading.
## - updated the docstring in min_fill_ordering. The tie-breaking rule should
##   choose the variable that comes first in the provided list of factors.
############################################################

import itertools
from bnetbase import Variable, Factor, BN
import csv


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''
    total_sum = sum(factor.values)
    normalized_values = []
    for value in factor.values:
        if total_sum == 0:
            normalized_values.append(0) 
        else:
            normalized_values.append(value/total_sum) 

    scope = factor.get_scope()
    domains = [variable.domain() for variable in scope] #[[1, 2, 3], ['a', 'b'], ['heavy', 'light']]
    domain_products = list(itertools.product(*domains))

    all_values = []
    for i in range(len(domain_products)):
        variable_values = list(domain_products[i]) #[1, 'a', 'heavy']
        variable_values.append(normalized_values[i])
        all_values.append(variable_values)

    new_factor = Factor(factor.name + "-normalized", scope)
    new_factor.add_values(all_values)
    return new_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    ''' 
    scope = factor.get_scope()
    variable_index = scope.index(variable)
    new_scope = scope.copy()
    new_scope.pop(variable_index)

    domains = [variable.domain() for variable in new_scope] #[[1, 2, 3], ['a', 'b'], ['heavy', 'light']]
    domain_products = itertools.product(*domains)

    all_values = []
    for product in domain_products:
        variable_values = list(product) #[1, 'a', 'heavy']
        variable_values.insert(variable_index, value)
        factor_value = factor.get_value(variable_values)
        variable_values.pop(variable_index)
        variable_values.append(factor_value)
        all_values.append(variable_values)
    
    new_factor = Factor(factor.name + "," + variable.name + "-restricted", new_scope)
    new_factor.add_values(all_values)
    return new_factor


def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''       
    scope = factor.get_scope()
    variable_index = scope.index(variable)
    new_scope = scope.copy()
    new_scope.pop(variable_index)

    domains = [variable.domain() for variable in new_scope] #[[1, 2, 3], ['a', 'b'], ['heavy', 'light']]
    domain_products = itertools.product(*domains)

    all_values = []
    for product in domain_products:
        variable_values = list(product) #[1, 'a', 'heavy']
        value = 0
        for variable_value in variable.domain():
            variable_values.insert(variable_index, variable_value)
            value += factor.get_value(variable_values)
            variable_values.pop(variable_index)

        variable_values.append(value)
        all_values.append(variable_values)
    
    new_factor = Factor(factor.name + "," + variable.name + "-summed", new_scope)
    new_factor.add_values(all_values)
    return new_factor


def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    ''' 
    new_scope, name = [], ''
    for factor in factor_list:
        name += factor.name + ',' 
        for variable in factor.get_scope():
            if variable not in new_scope:
                new_scope.append(variable)

    domains = [variable.domain() for variable in new_scope] #[[1, 2, 3], ['a', 'b'], ['heavy', 'light']]
    domain_products = itertools.product(*domains)

    all_values = []
    for product in domain_products:
        variable_values = list(product) #[1, 'a', 'heavy']
        value = 1
        for factor in factor_list:
            values = []
            for variable in factor.get_scope():
                for i in range(len(new_scope)):          
                    if variable == new_scope[i]:
                        values.append(variable_values[i])
            if values:
                value *= factor.get_value(values)

        variable_values.append(value)
        all_values.append(variable_values)

    new_factor = Factor(name + "-multiplied", new_scope)
    new_factor.add_values(all_values)
    return new_factor


def min_fill_ordering(factor_list, variable_query):
    '''
    This function implements The Min Fill Heuristic. We will use this heuristic to determine the order 
    to eliminate the hidden variables. The Min Fill Heuristic says to eliminate next the variable that 
    creates the factor of the smallest size. If there is a tie, choose the variable that comes first 
    in the provided order of factors in factor_list.

    The returned list is determined iteratively.
    First, determine the size of the resulting factor when eliminating each variable from the factor_list.
    The size of the resulting factor is the number of variables in the factor.
    Then the first variable in the returned list should be the variable that results in the factor 
    of the smallest size. If there is a tie, choose the variable that comes first in the provided order of 
    factors in factor_list. 
    Then repeat the process above to determine the second, third, ... variable in the returned list.

    Here is an example.
    Consider our complete Holmes network. Suppose that we are given a list of factors for the variables 
    in this order: P(E), P(B), P(A|B, E), P(G|A), and P(W|A). Assume that our query variable is Earthquake. 
    Among the other variables, which one should we eliminate first based on the Min Fill Heuristic? 

    - Eliminating B creates a factor of 2 variables (A and E).
    - Eliminating A creates a factor of 4 variables (E, B, G and W).
    - Eliminating G creates a factor of 1 variable (A).
    - Eliminating W creates a factor of 1 variable (A).

    In this case, G and W tie for the best variable to be eliminated first since eliminating each variable 
    creates a factor of 1 variable only. Based on our tie-breaking rule, we should choose G since it comes 
    before W in the list of factors provided.
    '''
    variables = []
    for factor in factor_list:
        for variable in factor.get_scope():
            if variable != variable_query and variable not in variables:
                variables.append(variable)

    sorted_variables = []
    factors = factor_list.copy()
    while variables:
        min_size = float('inf')
        min_variable, min_factor = None, None
        for variable in variables:
            test_factors = [factor for factor in factors if variable in factor.get_scope()]
            res_factor = sum_out(multiply(test_factors), variable)
            if len(res_factor.get_scope()) < min_size:
                min_size = len(res_factor.get_scope())
                min_variable = variable
                min_factor = res_factor
                print(min_variable, min_size)
        sorted_variables.append(min_variable)
        variables.remove(min_variable)
        factors = [factor for factor in factors if min_variable not in factor.get_scope()]
        factors.append(min_factor)

    print("=====")
    for v in sorted_variables:
        print(v, v.domain_size())
    return sorted_variables


def ve(bayes_net, var_query, varlist_evidence): 
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by varlist_evidence. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    '''
    remaining_factors = []
    for factor in bayes_net.factors():
        restricted_factor = factor
        for variable in restricted_factor.get_scope():
            if variable in varlist_evidence:
                restricted_factor = restrict(restricted_factor, variable, variable.get_evidence())
        remaining_factors.append(restricted_factor)
  
    for variable in min_fill_ordering(remaining_factors, var_query):
        variable_factors = [factor for factor in remaining_factors if variable in factor.get_scope()]
        res_factor = sum_out(multiply(variable_factors), variable)
        remaining_factors = [factor for factor in remaining_factors if factor not in variable_factors]
        remaining_factors.append(res_factor)
    
    multiplied_factor = multiply(remaining_factors)
    return normalize(multiplied_factor)


## The order of these domains is consistent with the order of the columns in the data set.
salary_variable_domains = {
"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
"Gender": ['Male', 'Female'],
"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
"Salary": ['<50K', '>=50K']
}

salary_variable=Variable("Salary", ['<50K', '>=50K'])

def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents 
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as 
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of 
    the dataset are the CLASS that we want to predict.

    Please name the factors as follows. If you don't follow these naming conventions, you will fail our tests.
    - The name of the Salary factor should be called "Salary" without the quotation marks.
    - The name of any other factor should be called "VariableName,Salary" without the quotation marks. 
      For example, the factor for Education should be called "Education,Salary".

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    # Variables
    work = Variable("Work", salary_variable_domains["Work"])
    education = Variable("Education", salary_variable_domains["Education"])
    marital_status = Variable("MaritalStatus", salary_variable_domains["MaritalStatus"])
    occupation = Variable("Occupation", salary_variable_domains["Occupation"])
    relationship = Variable("Relationship", salary_variable_domains["Relationship"])
    race = Variable("Race", salary_variable_domains["Race"])
    gender = Variable("Gender", salary_variable_domains["Gender"])
    country = Variable("Country", salary_variable_domains["Country"])

    # Factors
    work_factor = Factor("Work,Salary", [work, salary_variable])
    education_factor = Factor("Education,Salary", [education, salary_variable])
    marital_status_factor = Factor("MaritalStatus,Salary", [marital_status, salary_variable])
    occupation_factor = Factor("Occupation,Salary", [occupation, salary_variable])
    relationship_factor = Factor("Relationship,Salary", [relationship, salary_variable])
    race_factor = Factor("Race,Salary", [race, salary_variable])
    gender_factor = Factor("Gender,Salary", [gender, salary_variable])
    country_factor = Factor("Salary,Salary", [country, salary_variable])
    salary_factor = Factor("Salary", [salary_variable])

    # Variable Indices
    indices = {}
    for i in range(len(headers)):
        indices[headers[i]] = i
    
    # Salary Factor
    total_individuals = len(input_data)
    below, above = 0, 0
    for individual in input_data:
        if individual[indices['Salary']] == salary_variable.domain()[0]:
            below += 1
        else:
            above += 1
    salary_factor.add_values([[salary_variable.domain()[0], below/total_individuals], [salary_variable.domain()[1], above/total_individuals]])

    # Conditional Factors
    factors = [work_factor, education_factor, marital_status_factor, occupation_factor, 
               relationship_factor, race_factor, gender_factor, country_factor, salary_factor]
    for i in range(len(factors[:-1])):
        factor = factors[i]
        domains = [variable.domain() for variable in factor.get_scope()] #[['Male', 'Female'], ['<50K', '>=50K']]
        domain_products = itertools.product(*domains)

        all_values = []
        for product in domain_products:
            variable_values = list(product) #['Female', '>=50K']
            curr_value = variable_values[0]
            salary_value = variable_values[1]

            variable_count, salary_count = 0, 0
            for individual in input_data:
                if individual[indices['Salary']] == salary_value:
                    salary_count += 1
                    if individual[indices[factor.get_scope()[0].name]] == curr_value:
                        variable_count += 1
    
            probability = variable_count/salary_count if salary_count != 0 else 0
            variable_values.append(probability)
            all_values.append(variable_values)

        factor.add_values(all_values)

    variables = [work, education, marital_status, occupation, relationship, race, gender, country, salary_variable]
    return BN('Naive Bayes Model', variables, factors)


def explore(bayes_net, question):
    '''    
    Return a probability given a Naive Bayes Model and a question number 1-6. 
    
    The questions are below: 
    1. What percentage of the women in the test data set does our model predict having a salary >= $50K? 
    2. What percentage of the men in the test data set does our model predict having a salary >= $50K? 
    3. What percentage of the women in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    4. What percentage of the men in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    5. What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
    6. What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?

    @return a percentage (between 0 and 100)
    ''' 
    ### READ IN THE DATA
    input_data = []
    with open("data/adult-test.csv", newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)
    
    # Salary and Evidence Variables
    salary = bayes_net.get_variable('Salary')
    gender = bayes_net.get_variable('Gender')
    work = bayes_net.get_variable("Work")
    education = bayes_net.get_variable("Education")
    occupation = bayes_net.get_variable("Occupation")
    relationship = bayes_net.get_variable("Relationship")
    evidence_variables = [work, education, occupation, relationship]

    # Variable Indices
    indices = {}
    for i in range(len(headers)):
        indices[headers[i]] = i

    if question == 1:
        above, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Female':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K'])
                if probability > 0.5:
                    above += 1
                total += 1

        return 100 * (above/total)

    elif question == 2:
        above, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Male':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K'])
                if probability > 0.5:
                    above += 1
                total += 1

        return 100 * (above/total)
    
    elif question == 3:
        greater, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Female':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])
                gender.set_evidence('Female')

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K'])      
                gender_probability = ve(bayes_net, salary, evidence_variables + [gender]).get_value(['>=50K'])               
                if probability > gender_probability:
                    greater += 1
                total += 1

        return 100 * (greater/total)
    
    elif question == 4:
        greater, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Male':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])
                gender.set_evidence('Male')

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K']) 
                gender_probability = ve(bayes_net, salary, evidence_variables + [gender]).get_value(['>=50K'])
                if probability > gender_probability:
                    greater += 1
                total += 1

        return 100 * (greater/total)
    
    elif question == 5:
        actual, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Female':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K'])
                if probability > 0.5:
                    total += 1
                    if individual[indices['Salary']] == salary.domain()[1]:
                        actual += 1

        return 100 * (actual/total)
    
    elif question == 6:
        actual, total = 0, 0
        for individual in input_data:
            if individual[indices['Gender']] == 'Male':
                work.set_evidence(individual[indices['Work']])
                education.set_evidence(individual[indices['Education']])
                occupation.set_evidence(individual[indices['Occupation']])
                relationship.set_evidence(individual[indices['Relationship']])

                probability = ve(bayes_net, salary, evidence_variables).get_value(['>=50K'])
                if probability > 0.5:
                    total += 1
                    if individual[indices['Salary']] == salary.domain()[1]:
                        actual += 1
            
        return 100 * (actual/total)

# if __name__ == '__main__':
#     bn = naive_bayes_model('./adult-train.csv')
#     exp = explore(bn, 1)
#     print(exp)