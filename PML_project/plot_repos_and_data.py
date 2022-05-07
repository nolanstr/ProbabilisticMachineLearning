from bingo.evolutionary_optimizers.evolutionary_optimizer import \
                    load_evolutionary_optimizer_from_file as leoff
import numpy as np
import sympy as sy
import glob
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

import sys;sys.path.append('../../Tushar_help')

from util.ffs_and_data import get_ffs_w_xy
from util.get_best_ind_from_hof import get_best_from_hof
from util.cred_pred import *
from util.cred_pred_functions import *

data = np.load('noisy_data.npy')[1:,:]
yaml_file = 'gpsr_hyperparams1upd.yaml'
def return_hof(DIR):
    
    files = glob.glob(DIR + '/*.pkl')

    pickles = [leoff(FILE) for FILE in files]
    gens = [pickle[0].generational_age for pickle in pickles]
    idx = np.argmax(gens)

    final = pickles[idx]
    
    hof = final[0].hall_of_fame
    
    return hof

repos = ['sr2/gpsrUQ', 'sr1/gpsrUQ']
file_names = ['new_sr1', 'new_sr2']

for file_name, repo in zip(file_names, repos):

    hof = return_hof(repo)
    colors = [plt.cm.Pastel1(i) for i in range(len(hof))]
    clo, fbf, training_data = get_ffs_w_xy(yaml_file, data[:,0], data[:,1])
    
    fig, ax = plt.subplots(1)
    ax.scatter(training_data.x[:,0], training_data.y, c='k', label='Training Data')

    for model, color in zip(hof, colors):
        plot_model_on_fig(ax, model, training_data, fbf, color, 0, iters=20) 

    plt.tight_layout()
    plt.savefig(file_name, dpi=1000)

