import pandas as pd
import numpy as np
import random
import sys

sys.path.append('../omico')

import plot as pl
import table as tb
import session as es

sys.path.append('../dglv')

import sampler as dg

import sys

sys.path.append('./omico')

import plot as pl
import fit as ft
import analysis as an
import table as tb
import session as es

engine=  'kaiju'
phen = 'diagnosis'
database= 'RefSeq'
protocol= 'relative_abundance'
scale = 'relative'

if engine=='core-kaiju': cuts=np.arange(1,21)
else: cuts = [9e-06,0.0007]

print(engine,phen,database,protocol)

phen_cuts=cuts
nbins,digits=0,0
phenotype_map=pd.Series(dtype=str)
re_group={}

re_group = {'H':'H','UC':'U','CD':'U','IBS-C':'U','IBS-D':'U'}

Pheno = es.tables_session(engine=engine,
                      database=database,
                      PROJ_ROOT='../..',
                      cuts=phen_cuts,
                      protocol=protocol,
                      group=phen,
                      phenotype_map=phenotype_map,
                      nbins=nbins,digits=digits,
                      re_group=re_group
                      )

tH=Pheno['T']['H'][cuts[0]].form['relative']
tU=Pheno['T']['U'][cuts[0]].form['relative']

samples = list(set(tH.columns).union(tU.columns))
species = list(set(tH.index).union(tU.index))

tJ=pd.DataFrame(index=species,columns=samples,dtype=float)

for p in ['H','U']:
    
    for s in Pheno['T'][p][cuts[0]].samples:
        
        tJ[s]=Pheno['T'][p][cuts[0]].form['relative'][s]
        
tJ=tJ.fillna(0)

rH=len(Pheno['T']['H'][cuts[0]].samples)
rU=len(Pheno['T']['U'][cuts[0]].samples)

# ops with real data

replicates = 5000

H_rop = pd.DataFrame(columns=['h','qd','q0','K'],index=range(replicates),dtype=float)
U_rop = pd.DataFrame(columns=['h','qd','q0','K'],index=range(replicates),dtype=float)

for r in range(replicates):
    
    print(r)
    
    random.shuffle(samples)

    tH_random = tJ[ samples[:rH] ].values
    tU_random = tJ[ samples[rH:] ].values
    
    for t,op in zip([tH_random,tU_random],[H_rop,U_rop]):
        
        op.loc[r,'h']  = np.ma.array(t,mask= t<cuts[0] ).mean(axis=0).mean()
        op.loc[r,'qd'] = np.ma.array(t**2,mask= t<cuts[0] ).mean(axis=0).mean()
        op.loc[r,'q0'] = (np.ma.array(t,mask= t<cuts[0] ).mean(axis=0)**2).mean()
        op.loc[r,'K']  = t.max(axis=1).mean()

# ops with the data
H_op = pd.DataFrame(columns=['h','qd','q0','K'],index=range(replicates),dtype=float)
H_samples = Pheno['T']['H'][cuts[0]].samples

U_op = pd.DataFrame(columns=['h','qd','q0','K'],index=range(replicates),dtype=float)
U_samples = Pheno['T']['U'][cuts[0]].samples

for r in range(replicates):
    
    print(r)
    
    random.shuffle(H_samples)
    random.shuffle(U_samples)

    tH_random = tH[ H_samples[:int(0.9*rH)] ].values
    tU_random = tU[ U_samples[:int(0.9*rU)] ].values
    
    for t,op in zip([tH_random,tU_random],[H_op,U_op]):
        
        op.loc[r,'h']  = np.ma.array(t,mask= t<cuts[0] ).mean(axis=0).mean()
        op.loc[r,'qd'] = np.ma.array(t**2,mask= t<cuts[0] ).mean(axis=0).mean()
        op.loc[r,'q0'] = (np.ma.array(t,mask= t<cuts[0] ).mean(axis=0)**2).mean()
        op.loc[r,'K']  = t.max(axis=1).mean()

# prepare results for the plot
W=0.4

H_op['phenotype']='healthy'
H_op['label']='data'
H_op['x']=0

H_rop['phenotype']='healthy'
H_rop['label']='random'
H_rop['x']=W

U_op['phenotype']='diesase'
U_op['label']='data'
U_op['x']=1

U_rop['phenotype']='diesase'
U_rop['label']='random'
U_rop['x']=1+W

H_op.to_csv('../../data/OPs/gutH.csv')
H_rop.to_csv('../../data/OPs/gutH_random.csv')
U_op.to_csv('../../data/OPs/gutU.csv')
U_rop.to_csv('../../data/OPs/gutU_random.csv')

# prepare summary OP to use in optimization
H_summary=pd.Series(index=['h','qd','q0','K'],dtype=float)

H_summary['h'] = np.ma.array(tH,mask= tH<cuts[0] ).mean(axis=0).mean()
H_summary['qd'] = np.ma.array(tH**2,mask= tH<cuts[0] ).mean(axis=0).mean()
H_summary['q0'] = (np.ma.array(tH,mask= tH<cuts[0] ).mean(axis=0)**2).mean()
H_summary['K'] = tH.max(axis=1).mean()
H_summary.index=H_summary.index.rename('OP')
H_summary=H_summary.rename('Bootstrap value')

U_summary=pd.Series(index=['h','qd','q0','K'],dtype=float)

U_summary['h'] = np.ma.array(tU,mask= tU<cuts[0] ).mean(axis=0).mean()
U_summary['qd'] = np.ma.array(tU**2,mask= tU<cuts[0] ).mean(axis=0).mean()
U_summary['q0'] = (np.ma.array(tU,mask= tU<cuts[0] ).mean(axis=0)**2).mean()
U_summary['K'] = tU.max(axis=1).mean()
U_summary.index=U_summary.index.rename('OP')
U_summary=U_summary.rename('Bootstrap value')
H_summary.to_csv('../../data/OPs/gutH_summary.csv')
U_summary.to_csv('../../data/OPs/gutU_summary.csv')