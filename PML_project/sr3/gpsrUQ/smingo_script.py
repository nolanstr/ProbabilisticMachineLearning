import glob
import numpy as np
import sys 
import yaml
sys.path.append('../')
sys.path.append('../../')


from bingo_nolan_fork.bingo.local_optimizers.continuous_local_opt \
        import ContinuousLocalOptimization
from bingo_nolan_fork.bingo.symbolic_regression.explicit_regression \
        import ExplicitRegression, ExplicitTrainingData
from bingo_nolan_fork.bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo_nolan_fork.bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo_nolan_fork.bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo_nolan_fork.bingo.symbolic_regression.agraph.component_generator \
        import ComponentGenerator
from bingo_nolan_fork.bingo.evaluation.evaluation import Evaluation
from bingo_nolan_fork.bingo.stats.pareto_front import ParetoFront
from bingo_nolan_fork.bingo.evolutionary_optimizers.parallel_archipelago import \
                                ParallelArchipelago
from bingo_nolan_fork.bingo.evolutionary_optimizers.parallel_archipelago import \
                    load_parallel_archipelago_from_file

from bingo_nolan_fork.bingo.evolutionary_algorithms.generalized_crowding import \
                                GeneralizedCrowdingEA
from bingo_nolan_fork.bingo.evolutionary_optimizers.island import Island


from bingo_nolan_fork.bingo.symbolic_regression.bayes_fitness_function import \
                        BayesFitnessFunction

from yaml_extractor import *

def build_component_generator(training_data, operators):

    component_generator = ComponentGenerator(training_data.x.shape[1])
    for op in operators:
        component_generator.add_operator(int(op))

    return component_generator

def bridge(inputs):
    
    gparams, sparams, dsets = \
                    pull_data_from_yaml('../../gpsr_hyperparams1upd.yaml')

    data = np.load('../../noisy_data.npy')
    training_data = ExplicitTrainingData(x=data[:,0], y=data[:,1])

    component_generator = build_component_generator(training_data, gparams[9])
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    

    fitness = ExplicitRegression(training_data=training_data,
                        metric='mean absolute error')
    clo = ContinuousLocalOptimization(fitness, algorithm='lm')

    agraph_generator = AGraphGenerator(gparams[2], component_generator)

    fbf = BayesFitnessFunction(clo,
            num_particles=sparams[0], mcmc_steps=sparams[1])
    evaluator = Evaluation(fbf)#, redundant=True, multiprocess=1)

    ea = GeneralizedCrowdingEA(evaluator, crossover,
                mutation, gparams[7], gparams[8])
    
    pareto_front = ParetoFront(secondary_key = lambda ag: ag.get_complexity(), 
                            similarity_function=agraph_similarity)

    island = Island(ea, agraph_generator, gparams[0], hall_of_fame=pareto_front)
    
    parallel_archipelago = ParallelArchipelago(island, hall_of_fame=pareto_front)
    
    optim_result = parallel_archipelago.evolve_until_convergence(gparams[3],
            gparams[4], convergence_check_frequency=gparams[5],
                checkpoint_base_name='smingo')

def agraph_similarity(ag_1, ag_2):
    return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

if __name__ == "__main__":
    bridge(0)
