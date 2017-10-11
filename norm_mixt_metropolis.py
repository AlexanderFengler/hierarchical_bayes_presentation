#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:01:04 2017

@author: augustofasano
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import ceil

#functions
def mixt_sampling(delta,theta1,theta2):
    #sampling from mixture of two normals
    if np.random.uniform(low=0,high=1,size=1) <= delta:
        theta=theta1
    else:
        theta=theta2        
    return np.random.normal(theta[0], theta[1],size=1)

def mixt_density(x,delta,rv1,rv2):
    #density in x of the mixture of two rvs with mixing weight given by delta
    return delta*rv1.pdf(x)+(1-delta)*rv2.pdf(x)

#generate the data
nObs=100
delta=0.7
theta1=[7, 0.5]
theta2=[10, 0.5]

y=np.zeros((100,1))
np.random.seed(15)

for i in range(nObs):
    y[i]=mixt_sampling(delta,theta1,theta2)

#plot histogram with relative frequencies
plt.hist(y,normed=True)

##add plot of the density
x=np.linspace(np.min(y)-0.5,np.max(y)+0.5,200)
rv1=stats.norm(loc=theta1[0],scale=theta1[1])
rv2=stats.norm(loc=theta2[0],scale=theta2[1])

f=mixt_density(x,delta,rv1,rv2)

plt.plot(x,f)



#MCMC part
np.random.seed(16)
nSim=10000
delta_chain=np.zeros((nSim,2))
delta_chain[0,:]=0.5
candidate=np.zeros((1,2))
loglike_old=np.zeros((1,2))
loglike_candidate=np.zeros((1,2))
ratio=np.zeros((1,2))
for i in range(1,nSim):
    loglike_old=np.array([np.sum(np.log(mixt_density(y,delta_chain[i-1,0],rv1,rv2))), np.sum(np.log(mixt_density(y,delta_chain[i-1,1],rv1,rv2)))])
    
    candidate=np.array([float(np.random.beta(1,1,1)), float(np.random.beta(2,10,1))])
    loglike_candidate=np.array([np.sum(np.log(mixt_density(y,candidate[0],rv1,rv2))),np.sum(np.log(mixt_density(y,candidate[1],rv1,rv2)))])
    ratio=np.array([np.exp(loglike_candidate[0]-loglike_old[0]), np.exp(loglike_candidate[1]-loglike_old[1])])
    
    if np.random.uniform(0,1,1) <= ratio[0]:
        delta_chain[i,0]=candidate[0]
    else:
        delta_chain[i,0]=delta_chain[i-1,0]
        
    if np.random.uniform(0,1,1) <= ratio[1]:
        delta_chain[i,1]=candidate[1]
    else:
        delta_chain[i,1]=delta_chain[i-1,1]



plt.subplot(211)
plt.plot(delta_chain[:,0])
plt.title('Beta(1,1) candidate distribution')

plt.subplot(212)
plt.plot(delta_chain[:,1])
plt.title('Beta(2,10) candidate distribution')

plt.tight_layout()
plt.show


plt.subplot(211)
plt.hist(delta_chain[range(ceil(nSim/2)),0])
plt.title('Beta(1,1) candidate distribution')

plt.subplot(212)
plt.hist(delta_chain[range(ceil(nSim/2)),1])
plt.title('Beta(2,10) candidate distribution')

plt.tight_layout()
plt.show
