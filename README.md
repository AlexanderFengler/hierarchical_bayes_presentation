# hierarchical_bayes_presentation

### Gibbs sampler notes

To run the gibbs sampler, open python from within the `hierarchical_bayes_presentation` folder, and type the following commands in python

```python
import Hierarchical_normal_model_gibbs as hnmg
out = hnmg.main('support_files/hierarchical_normal_data.txt',
                 n_iter = 5000, 
                 n_chains = 10, 
                 save = True)
```
You will get a **results_gibbs** folder with **csv** files of names *gibbs_chain_i.csv* (*i*, number of the chain), that contains `n_iter` samples from overy chain for every parameters posterior distribution. 

### Pystan Notes

##### Getting Pystan
To run the hierarchical_model.py file you need to install pystan.
- Run `sudo xcodebuild -license accept` in the command line **first** to make sure you accepted the *xcode license agreement*, otherwise pystan doesn't install properly.
-  If you don't have *anaconda* run `pip install pystan` from the command line to get pystan.
-  If you have *anaconda* run `conda install -c conda-forge pystan` 

##### Running the files

To perform a run of the model and get samples from the chain in csv files, do the following.

1. Start python from within the `hierarchical_bayes_presentation` folder. 
2. Use the following python commands

```python
import Hierarchical_normal_model_stan as hnms
out = hnms.main('support_files/hierarchical_normal_data.txt', 
                              'support_files/hierarchical_normal_stan_model.txt', 
                              n_iter = 5000, 
                              n_chains = 10)
``` 
You will get an **results_stan** folder with **csv** files of names *stan_chain_i.csv* (*i*, number of the chain), that contains `n_iter // 2` samples from every chain for every parameter specified in the model. The number of samples is half the number of specified iterations, because by default **pystan** will use half of the speficied sample size as a **warm-up** samples to tune the adaptive sampling algorithm.

### Timings notes

To compare timings of the stan and gibbs samplers, run the following code in python

```python
import Hierarchical_normal_sampler_timings as hnst
out = hnst.create_timings_table(n_iters = [100, 1000, 10000])
```
An **html** table will be created in the main folder which illustrates the timing comparisons. Under `n_iters` an arbitrary list of *sample-lengths* can be specified. 

**Warning:** For 10000 iterations the gibbs sampler uses already approximately a minute of time, and time grows **O(n_iter)**.
