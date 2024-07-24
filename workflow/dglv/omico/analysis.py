import pandas as pd
import numpy as np
import scipy as sp
import os
import random
import math
from string import ascii_uppercase
from sklearn.metrics import mutual_info_score
from sklearn.metrics import mean_squared_error, r2_score

import fit as ft

def NimwegenLaws(T,observable,log_x=True,log_y=True,ensemble_size=10,train_percentage=0.9,confidence=0.95):
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
    fields=list(ef.linear_bootstrap(x,x,confidence=confidence).keys())
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
                r=ef.linear_bootstrap(x=x,y=y,
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

def TaylorLaws(T,observable,log_x=True,log_y=True,ensemble_size=10,train_percentage=0.9,confidence=0.95):
    
    try: 
        L = T.observables['size']
    except KeyError: 
        print(' * * > size partition absent ')
        
    # initialize the report with the fields of the fit
    x=np.logspace(-1,1)
    fields=list(ef.linear_bootstrap(x,x,confidence=confidence).keys())
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
                r=ef.linear_bootstrap(x=x,y=y,
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

def compare_neutral_curves(A_data,A_model,delta,n_bins):
    
    bins=np.linspace(A_data['x'].min()-delta,A_data['x'].max()+delta,n_bins)
    
    x_group_data=pd.cut( A_data['x'],bins=bins )
    x_group_model=pd.cut( A_model['x'],bins=bins )

    C_data  = pd.Series(dtype=float)
    C_model = pd.Series(dtype=float)

    #if x_group_data.values.categories == x_group_model.values.categories:
        
    for m in x_group_data.values.categories:

        cl = x_group_data[x_group_data==m].index
        C_data[m.mid] = A_data['y'][cl].mean()
            
        cl = x_group_model[x_group_model==m].index
        C_model[m.mid] = A_model['y'][cl].mean()
            

    idx = set(C_data.dropna().index).intersection(C_model.dropna().index)
    C_data=C_data[idx].sort_index()
    C_model=C_model[idx].sort_index()
    
    return r2_score(C_data,C_model)

def jaccard_similarity(D,to_bool=False):
    
    if to_bool==True:

        D=1*(D>0)
        
    one = pd.DataFrame(index=D.index,columns=D.columns,data=np.ones(D.shape))

    K=((D.T).dot(D))/( (one.T).dot(D) + (D.T).dot(one) - (D.T).dot(D) )
    
    K = K.where(np.triu(np.ones(K.shape),k=1).astype(np.bool))

    return K

def RarefactionTable(T,e=10,report=False,save=True,file=''):
    
    R=pd.DataFrame(index=range(e),columns=range(1,len(T.samples)+1))
    
    G=T.form['binary']
    samples = list(G.columns)
    
    for i in R.columns:
        
        #print(i,'> ',end='')
        for j in R.index:
            
            random.shuffle(samples)
            idx = samples[:i]
            
            g=G[ idx ]
            g=g.sum(axis=1)
            g=g[g>0]
            R.loc[j,i]=g.shape[0]
        
    if save==True: R.to_csv(os.path.join(file))
      
    if report==True: return R
            