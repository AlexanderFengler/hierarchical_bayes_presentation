import Hierarchical_normal_model_stan
import Hierarchical_normal_model_gibbs
import pandas as pd
import numpy as np
import decimal
import os



def get_timings(n_iter = 1000, n_chains = 1, runs = 10):
    stan_timings = list()
    gibbs_timings = list()
    for i in range(runs):
        stan_timings.append(Hierarchical_normal_model_stan.main(data_file = 'support_files/hierarchical_normal_data.txt',
                                           model_file = 'support_files/hierarchical_normal_stan_model.txt',
                                           n_iter = n_iter,
                                           n_chains = n_chains,
                                           performance_test = True))
        gibbs_timings.append(Hierarchical_normal_model_gibbs.main(data_file = 'support_files/hierarchical_normal_data.txt',
                                           n_iter = n_iter, n_chains = n_chains, save = False, performance_test = True))
    return stan_timings, gibbs_timings

def create_timings_table(n_iters = [100, 1000, 10000]):
    df = pd.DataFrame()
    df[''] = ['stan', 'gibbs']
    for i in n_iters:
        stan_timings, gibbs_timings = get_timings(n_iter = i)
        stan_mean = np.mean(stan_timings)
        gibbs_mean = np.mean(gibbs_timings)
        df[str(i)] = [stan_mean, gibbs_mean]
    df.set_index('')
    decimal.getcontext().prec = 3
    df.to_html('timings_table.html',float_format = lambda x: '%10.3f' % x)
    return(df)
