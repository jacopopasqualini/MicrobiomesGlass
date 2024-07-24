import pandas as pd
import numpy as np

import scipy as sp
from scipy import stats 

import os
import random

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def chi_squared(o,e):
    return np.sum( (e-o)**2/o )

def regression_bootstrap(x,y,model=linear_model.LinearRegression(),ensemble_size=10,train_percentage=0.9,confidence=0.99):
    
    S = x.shape[0]

    train_size = int(S*train_percentage); test_size = S - train_size
    
    ensemble = np.zeros((ensemble_size,2))
    err = np.zeros(ensemble_size)
    
    # fit model with boostrap procedure
    #g,h = np.log(x), np.log(y)
    g,h=x,y
    
    for t in range(ensemble_size):
    
        index=list(range(S)); random.shuffle( index )
        
        train_index = index[:train_size]
        x_train = g[train_index].reshape((train_size,1))
        y_train = h[train_index].ravel()#.reshape((train_size,1))
        
        test_index = index[train_size:]
        x_test = g[test_index].reshape((test_size,1))
        y_test = h[test_index].reshape((test_size,1))
        
        model.fit(X=x_train,y=y_train)
        model.get_params()
        
        y_pred = model.predict(x_test)
        
        ensemble[t,0] = model.coef_[0]
        ensemble[t,1] = model.intercept_
        
        err[t] = mean_squared_error(y_test, y_pred)
    
    del g, h
    
    slopes, intercepts = ensemble[:,0], ensemble[:,1]
    
    # weighted averaging
    Z=np.sum(1/err); 
    weights = (1/err)/Z
    W = [0,0]
    
    W[0], W[1] = np.dot(weights,slopes), np.dot(weights,intercepts)
    
    y_bag = W[1]+W[0]*x
    err = np.sqrt( np.sum((y-y_bag)**2)/S )
    
    try:
        R2=r2_score(y_bag,y)
    except ValueError:
        R2 = 0
    
    CI_inter = [ np.percentile(intercepts,confidence/100),np.percentile(intercepts,100-confidence/100) ]
    CI_slope = [ np.percentile(slopes,confidence/100),np.percentile(slopes,100-confidence/100) ]
    
    c= str(confidence)

    K = {'intercept':W[1],'slope':W[0],
         c+'_l_CI_intercept':CI_inter[0],c+'_r_CI_intercept':CI_inter[1],
         c+'_l_CI_slope':CI_slope[0],c+'_r_CI_slope':CI_slope[1],
         'R2':R2, 'L2_error':err } 

    return K

def pdf_bootstrap(samples,model,train_percentage,ensemble_size,confidence):
        
    #samples = M.copy()
    S=samples.shape[0]
    
    n_params=len(model.fit(np.random.uniform(0,1,10), floc=0))
    
    train_size = int(S*train_percentage); test_size = S - train_size
    
    ensemble = np.zeros((ensemble_size,n_params))
    err = np.zeros(ensemble_size)
    canonical_weights = np.zeros(ensemble_size)
    
    unique, counts = np.unique(samples, return_counts=True)
    empirical_freqs = counts/np.sum(counts)
    
    for t in range(ensemble_size):
        
        index=list(range(S)); random.shuffle( index )

        train_index=index[:train_size]
        x_train=samples[train_index];
        
        params = model.fit(x_train,floc=0)
        estimated_freqs = model.pdf( unique,*params)
        err[t]=chi_squared(empirical_freqs,estimated_freqs)
        
        for i in range(n_params):
            ensemble[t,i] = params[i] 

    weights = np.ones(ensemble_size)#(1/err)/(np.sum(1/err))
    W=np.mean(ensemble*(weights.reshape(ensemble_size,1)),axis=0)
    CI=[]
    
    results={}
    P={}
    for i in range(n_params):
        
        p=ensemble[:,i]
        CI = [np.percentile(p,confidence/100),np.percentile(p,100-confidence/100)] 
        results['CI_' + str(i+1)] = CI
        P['p_' + str(i+1)] = W[i]
    
    results['parameters']=P    
    results['chi']= chi_squared(empirical_freqs,estimated_freqs)
        
    return results

def fit_taxa(mu_taxa,sigma_taxa,verbose=True,compress=True):
    
    mad_boots = pdf_bootstrap(samples=mu_taxa,
                               model=stats.lognorm,train_percentage=0.9,
                               ensemble_size=100,confidence=0.95)

    taylor_params= powerlaw_fit(x=mu_taxa.values,y=sigma_taxa.values,
                                  ensemble_size=100,train_percentage=0.9,
                                  confidence=0.95)
    
    if verbose==True:
        print('>>> MAD:')
        print(mad_boots)
        print('>>> TAYLOR')
        print(taylor_params)
    
    if compress==False:
        return mad_params, taylor_params
    else:
        cricket={}
        mad_params=mad_boots['parameters']
        names=['shape','loc','scale']

        for n,p in zip(names,mad_params.keys()):
            mad_params[n]=mad_params[p]
            del mad_params[p]
        cricket['mad']=mad_params
        cricket['amplitude']=np.exp(taylor_params['intercept'])
        cricket['exponent']=taylor_params['slope']
        
        return cricket

def NimwegenLaws(T,observable,log_x=True,log_y=True,model=linear_model.LinearRegression(),ensemble_size=10,train_percentage=0.9,confidence=0.95):
    # aggiusta nomi e la intercep
    try: 
        L = T.observables['size']
    except KeyError: 
        print(' * * > size partition absent ')
        
    try: 
        L = L[observable]
    except: 
        print(' * * > observable not available ')
    
    # initialize the report with the fields of the fit
    x=np.logspace(-1,1)
    fields=list(regression_bootstrap(x,x,model=model,confidence=confidence).keys())
    Nimwegen=pd.DataFrame(columns=fields+['xy_correlation','lin_correlation'])
    
    # for each component fit the nimwegen laws
    for c in T.components:

        # the mass is the index, the 
        u=L.loc[c]
        u=u[u>0]

        x=u.index.values.astype(float)
        y=u.values
        
        if log_x==True: x = np.log(x)
        else: pass

        if log_y==True: y = np.log(y)
        else: pass

        if x.shape[0]>1:
            
            try:
                r=regression_bootstrap(x=x,y=y,model=model,
                                  ensemble_size=ensemble_size,
                                  train_percentage=train_percentage,
                                  confidence=confidence)

                nim=pd.DataFrame(r,index=pd.Index([c],name=T.annotation))
                Nimwegen=Nimwegen.append(nim)
                Nimwegen.loc[c,'xy_correlation']=np.corrcoef(x,y)[0,1]
                mod = r['intercept']+r['slope']*x
                Nimwegen.loc[c,'lin_correlation']=np.corrcoef(mod,y)[0,1]

            except RuntimeWarning:
                print(f' * * > {c}: not provided')
        else:
            print(f' * * > {c} : not provided, too few data points')
            
    Nimwegen.index = Nimwegen.index.rename(T.annotation)        
    Nimwegen=Nimwegen.sort_values('slope',ascending=False)
    
    return Nimwegen

def TaylorLaws(T,observable,model=linear_model.LinearRegression(),log_x=True,log_y=True,ensemble_size=10,train_percentage=0.9,confidence=0.95):
    
    try: 
        L = T.observables['size']
    except KeyError: 
        print(' * * > size partition absent ')
        
    # initialize the report with the fields of the fit
    x=np.logspace(-1,1)
    fields=list(regression_bootstrap(x,x,model=model,confidence=confidence).keys())
    Taylor=pd.DataFrame(columns=fields+['xy_correlation','mod_correlation'])
        
    # for each component fit the nimwegen laws
    
    for c in T.components:

        # the mass is the index, the 
        x=L[f'{observable} mean'].loc[c]
        y=L[f'{observable} var'].loc[c]
        
        w = set( (x[x>0]).index ).intersection( (y[y>0]).index )
        x,y=x[w].values.astype(float),y[w].values.astype(float)
        
        if log_x==True: x = np.log(x)
        else: pass

        if log_y==True: y = np.log(y)
        else: pass
    
        if x.shape[0]>1:
            
            try:
                r=regression_bootstrap(x=x,y=y,model=model,
                                      ensemble_size=ensemble_size,
                                      train_percentage=train_percentage,
                                      confidence=confidence)
                
                tay=pd.DataFrame(r,index=pd.Index([c],name=T.annotation))
                
                Taylor=Taylor.append(tay)
                Taylor.loc[c,'xy_correlation']=np.corrcoef(x,y)[0,1]
                #print(r['intercept'],r['slope'])
                mod = r['intercept']+r['slope']*x
                Taylor.loc[c,'mod_correlation']=np.corrcoef(mod,y)[0,1]
                
            except RuntimeWarning:
                print(f' * * > {c}: not provided')
                
        else:
            print(f' * * > {c} : not provided, too few data points')
            
             
    Taylor=Taylor.sort_values('slope',ascending=False)
    Taylor.index=Taylor.index.rename(T.annotation)
    
    return Taylor 