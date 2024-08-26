#!/usr/bin/env python
# coding: utf-8

# In[33]:


'''Import libraries and modules'''
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


'''RECURRING FUNCTIONS'''

'''Choose next stochastic time step (tau) and reaction (mu)'''
def choose_tau_mu(a0, a):
    
    # URN generator
    r1, r2 = np.random.random(2)
    
    # choose next time lag from r1
    tau = -1/a0*np.log(r1)
    
    # choose next rxn index mu ∈ {1,...,M} from r2
    mu = 0
    d = r2*a0 - a[mu]
    
    while d > 0:
        mu += 1
        d -= a[mu]
    
    return tau, mu

'''Calculate binomial coefficient for propensity h'''
def binom(n,k):
    
    b = [0]*(n+1)
    b[0] = 1
    for i in range(1,n+1):
        b[i] = 1
        j = i - 1
        while j>0:
            b[j] += b[j-1]
            j -= 1
    return b[k]

'''SIMULATION FUNCTION'''
def gillespie(Xinit, rates, stoch_react, stoch_prod, tmax, max_iterations):
    
    # define system stoichiometry as num_rxns by num_species (MxN) array
    stoch = stoch_react + stoch_prod
    s = np.shape(stoch)
    M = s[0]
    N = s[1]
    
    # initialize current time, current species numbers, time/rxn counters
    t = 0
    species = Xinit
    t_count = 0
    iterations = 0
    
    # initialize arrays to store time and species states
    largenum = 2*max_iterations
    store_t = np.zeros((largenum, 1))
    store_X = np.zeros((largenum, N))
    store_Rnum = np.zeros((largenum, 1))
    
    # update current time and species states
    store_t[t_count] = t # stores initial time in first entry of 1D time array
    store_X[t_count,:] = species # stores Xinit in first row of species states
    
    # main loop
    while t < tmax:
        
        # initialize rxn probs array
        a = np.ones((M,1))
        
        # calculate propensities
        for i in range(M):
            hi = 1
            for j in range(len(Xinit)):
                
                # check if reactant j is involved in rxn i
                if stoch_react[i,j] == 0:
                    continue
                else:
                    # check if reactant has remaining molecules
                    if species[j] <= 0:
                        hi = 0
                        continue
                    else:
                        hi = hi*binom(species[j], np.absolute(stoch_react[i,j]))*factorial(np.absolute(stoch_react[i,j]))
            
            
            a[i] = hi*rates[i]
        
        # normalize probabilities
        a0 = sum(a)
        
        # choose next time lag (tau) and rxn (mu)
        tau, mu = choose_tau_mu(a0, a)
        
        # update time and species states
        t += tau
        species += np.transpose(stoch[mu,:])
        
        # update counters
        t_count += 1
        iterations += 1
        
        # store updated states
        store_t[t_count] = t
        store_X[t_count,:] = species
        store_Rnum[t_count] = mu
        
    # store output
    store_t = store_t[:t_count]
    store_X = store_X[:t_count,:]
    store_Rnum = store_Rnum[:t_count]
    
    return store_t, store_X, store_Rnum


