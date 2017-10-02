# hierarchical_bayes_presentation

### Pystan Notes

##### Getting Pystan
To run the hierarchical_model.py file you need to install pystan.
- Run `sudo xcodebuild -license accept` in the command line **first** to make sure you accepted the *xcode license agreement*, otherwise pystan doesn't install properly.
-  If you don't have *anaconda* run `pip install pystan` from the command line to get pystan.
-  If you have *anaconda* run `conda install -c conda-forge pystan` 

##### Running the files

To perform a run of the model and get samples from the chain in csv files, do the following.

1. Start python from the `pystan_version` folder. 
2. Use the following python commands
```python
import hierarchical_model
out = hierarchical_model.main('hierarchical_normal_data.txt', 
                              'hierarchical_model.txt', 
                              n_samples = 5000, 
                              n_chains = 10, 
                              n_warmup = 0)
``` 
You will get an **output** folder with `n_chains` **csv** files, that contains `n_samples-n_warmup` samples from every chain for every parameter specified in the model.
