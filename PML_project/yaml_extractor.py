import yaml
import numpy as np

'''
gpsr hyperparams:

    gparams = [pop_size(int), offspring_size(int), stack_size(int), 
    max_generations(int), fitness_threshold(float), check_frequency(int),
    min_generations(int), crossover_probability(float),
    mutation_probability(float), mathematical_opterators(list of ints)]

smc hyperparams:

    sparams = [particles(int), smc_steps(int), mcmc_steps(int)]

data sets:

    dsets = [index of datasets(list)]
    this will be passed to another function)

'''


def pull_data_from_yaml(file_name):
    
    f = open(file_name)
    yfile = yaml.load(f, Loader=yaml.FullLoader)
    
    gparams = [val for val in yfile['gpsr'].values()]
    sparams = [val for val in yfile['smc'].values()]
    dsets = yfile['datasets'] 

    return gparams, sparams, dsets

