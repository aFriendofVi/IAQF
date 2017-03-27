
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model as lm
import math
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import operator
from itertools import product


# In[266]:

def MarkovianEM(data,target,N_state=2,itemax = 1000,llambda = 0):
    df = data.copy()

    # Choose target label to model
#     target = ['ChangePct']

    # Set number of states
#     N_state = 5

    # Set number of dimension
    D = len(target)

    # Initialize transition matrix
    P = np.identity(N_state)

    # Initialize Gaussian distribution for each state
    MU = np.random.rand(N_state,D)
    SIGMA = {}
    for i in range(N_state):
        aRandomMatrix = np.random.rand(D,D)
        SIGMA[i] = (np.dot(aRandomMatrix,aRandomMatrix.transpose())+np.identity(D))*100

    likelihoodList =[]
    
    # Iteratively solve the problem
    ite = 0
#     for ite in range(itemax):
    while True:
        
        #show progress
        if ite%100 == 99:
            print(str(100*(ite+1)/itemax)+'%')

        # initialize psi and responsibility for the first iteration
        if ite==0:
            psiList = [];
            rList = [];

        # Loop through every state
        for i in range(N_state):
            df['psi'+str(i)] = 0.00
            # compute the density of each data point for every Gaussian core
            df['psi'+str(i)] = st.multivariate_normal.pdf(df[target].values,mean = MU[i,:],cov = SIGMA[i])
            if ite==0:
                psiList.append('psi'+str(i))

        # Initialize the responsibility of each data point for every Gaussian core
        if ite==0:
            for i in range(N_state):    
                df['r'+str(i)] = df['psi'+str(i)]/df[psiList].sum(axis = 1)
                if ite==0:
                    rList.append('r'+str(i))

        df[rList].values[0,:] = df[psiList].values[0,:]/np.sum(df[psiList].values[0,:])
        
        # E(Expectation) Step:
        ## Update the Likelihood condition on the responsibility of the previous data point
        conditionalLikelihood = np.dot(df[rList].values[0:-1,],P)*df[psiList].values[1:]
        ## Update responsibility using the updated likelihood
        r = conditionalLikelihood/np.sum(conditionalLikelihood,axis =1)[:,np.newaxis]
        df.loc[1:,rList] = r
        
        # L(Likelihood) Step:
        ## Update the mean of the Gaussian Distribution for each state
        MU = np.dot(df[rList].values.transpose(),df[target].values)/(np.sum(df[rList].values,axis=0)[:,np.newaxis])
        
        likelihoodList.append(np.sum(np.multiply(np.exp(conditionalLikelihood),df[rList].values[1:,:])))
        
        ## Update the cov matrix of the Gaussian Distribution for each state
        for i in range(N_state):
            label = 'r'+str(i)
            Q = np.diag(df[label]/np.sum(df[label]))
            error = df[target].values-MU[i]
            SIGMA[i] = np.dot(np.squeeze(np.dot(Q,error).transpose()),np.squeeze(error))+np.identity(D)*llambda
        
        # Recalculate the transition matrix based on the updated responsibility
        for j,k in product(range(N_state),range(N_state)):
            P[j,k] = np.sum(df[['r'+str(j)]].values[0:-1]*df[['r'+str(k)]].values[1:],axis = 0)/np.sum(df[['r'+str(j)]].values[0:-1],axis = 0)
    
        #if ite >=10:
            #if (likelihoodList[len(likelihoodList)-1]-likelihoodList[-1])<0.000001:
                #break
        
        if ite>=itemax:
            break
        ite = ite +1
                           
    return df, MU, SIGMA, P, likelihoodList, rList




