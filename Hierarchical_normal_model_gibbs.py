import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import pylab
import pandas as pd
from pandas import DataFrame
from timeit import default_timer as timer
import os

np.random.seed(123)

def read_in_data_gibbs(file_name):
    data = dict()
    with open(file_name) as f:
        content = f.readlines()
    lines = [x.rstrip('\n') for x in content]
    for i in range(len(lines)):
        lines[i] = lines[i].split()
        if i <= 1:
            lines[i][1:] = [int(x) for x in lines[i][1:]]
            data[lines[i][0]] = lines[i][1:]
        else:
            lines[i][1:] = [float(x) for x in lines[i][1:]]
            data[lines[i][0]] = lines[i][1:]

    del data['J']
    del data['N']
    gibbsdat = []
    for i in list(data):
        gibbsdat.append(data[i])
    return gibbsdat

def theta_posterior(j , mi, sigma2, tau2, data):
    """ Evaluates the posterior distribution for the theta parameter of a group."""

    num = 1/tau2 * mi + len(data[j])/(sigma2)* np.mean(data[j])
    den = 1/tau2 + len(data[j])/sigma2
    thetajhat = num/den
    varhat = 1/den

    return stats.norm(thetajhat, np.sqrt(varhat))

def mi_posterior(theta, tau2):
    """ Evaluates the posterior distribution for the mi parameter"""
    return stats.norm(np.mean(theta), np.sqrt(tau2/len(theta)))

def sigma2_posterior(theta, data):
    """ Evaluates the posterior distribution for the sigma^2 parameter"""

    n_vec = [len(group) for group in data]
    n = sum(n_vec)
    n_groups = len(data)
    a = [np.dot(data[i]-theta[i],data[i]-theta[i]) for i in range(n_groups)]
    sigma2hat = sum(a)/n
    return stats.invgamma(a = n / 2, scale = n * sigma2hat /2)

def tau2_posterior(theta, mi):
    """ Evaluates the posterior distribution for the tau^2 parameter"""

    J = len(theta)
    theta = np.array(theta)
    tau2hat = np.dot(theta - mi, theta - mi) / (J-1)
    return stats.invgamma(a = (J-1)/2, scale = (J-1)* tau2hat /2)


# In[10]:


def set_starting_points(data, nchain):
    """ Sets the initial values for the theta parameter.

    The values are chosen by randomly chosing a value from the data.
    """

    starting_points = []
    for i in range(nchain):
        random_start = []
        for i in range(len(data)):
            random_start.append(np.random.choice(data[i]))
        starting_points.append( random_start)
    return starting_points



# In[14]:


def Gibbs_sampler(data, save = False, iterations = 1000, nchain = 10,  dirname = 'results_gibbs'):
    """ Approximates the target distribution through the Gibbs sampler.

    The underlying model assumes the presence of J groups whose distribution is assumed to be
    normal with same variance (sigma2_j = sigma2 for j=1..J) and different means (mi_j).
    The mean parameters (mi_j for j=1..J) are assumed to follow a normal distribution with
    mean mi and variance tau2.
    A uniform prior distribution is assumed for the joint distribution of
    (mi, log(sigma), log(tau)).

    The function returns a dataframe obtained by concatenating the second half of the
    values of each chain. Optionally the values for each chain are saved in csv files.
    """

    ngroups = len(data)
    starting_points = set_starting_points(data,nchain)
    colnames = ['theta' + str(i+1) for i in range(ngroups)] + ['mi', 'sigma', 'tau']

    mix_chain = DataFrame(columns = colnames)

    for iter_chain in range(nchain):

        theta = starting_points[iter_chain]
        chain = DataFrame(columns = colnames)

        mi = np.mean(theta)
        sigma2 = sigma2_posterior(theta, data).rvs(size=1)[0]
        tau2 = tau2_posterior(theta, mi).rvs(size=1)[0]

        chain.loc[0] = theta + [mi, np.sqrt(sigma2), np.sqrt(tau2)]

        for i in range(iterations-1):
            #if i % 50 == 0:
            #    print('i am a sample from gibbs')

            theta = [theta_posterior(j , mi, sigma2, tau2, data).rvs(size=1)[0] for j in range(len(theta))]
            mi = mi_posterior(theta,tau2).rvs(size=1)[0]
            tau2 = tau2_posterior(theta, mi).rvs(size = 1)[0]
            sigma2 = sigma2_posterior(theta, data).rvs(size = 1)[0]

            chain.loc[i+1] = theta + [mi, np.sqrt(sigma2), np.sqrt(tau2)]

        if save:
            current_dir = os.path.dirname('__file__')
            if not os.path.exists(os.path.join(current_dir,dirname)):
                os.makedirs(dirname)
            filename = os.path.join(current_dir, dirname, 'gibbs_chain'+str(iter_chain+1)+'.csv')
            chain.to_csv(filename)

        mix_chain = pd.concat([mix_chain, chain.tail(int(iterations/2))]).reset_index(drop=True)

    return mix_chain

def statistics(dataframe):
    """ Evaluates main quantiles for each column in the dataframe."""

    return dataframe.quantile([.05, .25, .5, .75, .95]).transpose()

def main(data_file = 'support_files/hierarchical_normal_data.txt', n_iter = 1000, n_chains = 10, save = False, performance_test = False):
    # full run of the model with output in csv file named chain_[*].csv
    data = read_in_data_gibbs(data_file)
    if not performance_test:
        return Gibbs_sampler(data = data, save = save, iterations = n_iter, nchain = n_chains)
    elif performance_test:
        start = timer()
        Gibbs_sampler(data = data, save = save, iterations = n_iter, nchain = n_chains)
        end = timer()
        return (end - start)
