import os
import pystan
import pickle
from hashlib import md5
import numpy as np

# to run the model run the main() function like shown below
# run: test = main('hierarchical_normal_data.txt', 'hierarchical_model.txt', 5000, 10)
# note: use help(test) to get information about the file you generated

# relevant filenames: model_code --> hierarchical_model.txt, data --> hierarchical_normal_data.txt
def read_in_data(file_name):
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
    return data

def read_in_model_code(file_name):
    # read in model code
    f = open(file_name, 'r+')
    model_code = f.read()
    return model_code

def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    # this function makes sure to reuse models so we don't have to recompile c everytime
    code_hash = md5(model_code.encode('ascii')).hexdigest() # give model some id derived from model code
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb')) # try to load model if it has been stored already
    except:
        sm = pystan.StanModel(model_code=model_code) # if model has not been stored compile it
        with open(cache_fn, 'wb') as f: # and save it for future use
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel") # if we reused the model print some message
    return sm


def initialize_model(model_code):
    sm = StanModel_cache(model_code = model_code)
    return sm

def run_model(stan_model, data, n_iter, n_chains, n_warmup):
    fit = stan_model.sampling(data = data, iter = n_iter, chains = n_chains, warmup = n_warmup)
    return fit

def write_samples_to_csv(fit_object):
    # create path if it doesn't exist
    mypath = 'output'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    # get first line of csv file as string (headers)
    fitdict = fit_object.extract(permuted = False)
    my_str = ''
    for i in range(len(fit_object.sim['fnames_oi'][:-1])):
        my_str += fit_object.sim['fnames_oi'][i] + ', '

    my_str = my_str.rstrip(', ')
    my_str += ' \n'

    # write samples to csv
    for i in range(np.shape(fitdict)[1]):
        with open('output/chain_' + str(i + 1) + '.csv', 'wb') as f:
            f.write(str.encode(my_str))
            np.savetxt(f, fitdict[:,i,:-1], delimiter=",", fmt = '%1.8f')
    return

def main(data_file, model_file, n_iter, n_chains, n_warmup):
    # full run of the model with output in csv file named chain_[*].csv
    data = read_in_data(data_file)
    model = read_in_model_code(model_file)
    sm = initialize_model(model)
    fit = run_model(sm, data, n_iter, n_chains, n_warmup)
    write_samples_to_csv(fit)
    return fit
