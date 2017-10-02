# hierarchical_bayes_presentation

### Pystan Notes

##### Getting Pystan
To run the hierarchical_model.py file you need to install pystan.
Run `pip install pystan` from the command line to get pystan. (*Note:* Run `sudo xcodebuild -license accept` in the command line **first** to make sure you accepted the *xcode license agreement*, otherwise your installation of pystan will not be functional)

##### Running the files

To perform a run of the model and get samples from the chain in csv files, do the following.
1. Start python from the `pystan_version` folder. 
2. Use the following python commands
```python
import hierarchical_model
``` 
