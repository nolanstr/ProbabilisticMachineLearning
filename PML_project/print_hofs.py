from bingo.evolutionary_optimizers.evolutionary_optimizer import \
                    load_evolutionary_optimizer_from_file as leoff
import numpy as np
import sympy as sy
import glob

def print_hof(DIR):
    
    files = glob.glob(DIR + '/*.pkl')

    pickles = [leoff(FILE) for FILE in files]
    gens = [pickle[0].generational_age for pickle in pickles]
    idx = np.argmax(gens)

    final = pickles[idx]
    
    hof = final[0].hall_of_fame
    for ind in hof:
        import pdb;pdb.set_trace()
        print(f'Fitness NMLL = {ind.fitness}')
        eq_str = ind.get_formatted_string("console")
        try:
            X_0=sy.symbols('X_0')
            ind_str = sy.expand(eq_str.replace(")(", ")*("))
            ind_str = sy.nsimplify(ind_str,tolerance=1e-5, rational=True)
        except:
            ind_str = eq_str
            pass
        print(f"Complexity: {ind.get_complexity()}\nfit: {ind.fitness} \
                \neqn: {ind_str}")
        print()
    import pdb;pdb.set_trace()

repos= ['sr1/gpsrUQ', 'sr2/gpsrUQ']
for repo in repos:
    print_hof(repo)
