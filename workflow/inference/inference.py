from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import IntegrationWarning

from concurrent.futures import ThreadPoolExecutor, TimeoutError, ProcessPoolExecutor

from pathlib import Path
from collections import OrderedDict
import concurrent.futures
import time
import numpy as np
import pandas as pd
import subprocess
import warnings
import os
import logging
import argparse
import opgd as og

from threading import Thread

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)
logging.basicConfig(level=logging.INFO)

class OPSearch():

    def __init__(self,pi,bounds0,r_dir,biome,search=4):

        print(30*'-')
        print(8*'-','DGLV OPTIMIZATION')
        print(30*'-')

        # recast the data before feeding them to the objective function
        self.pi = pi # parameters from the data 
        self.pi = {key:self.pi[key] for key in ['h', 'q0', 'qd', 'K', 'rho'] } 
        
        self.bounds0 = bounds0 # initial search buonds
        self.search = search # how many optimizations to perform
        self.theta_names = ['mu','lam','beta','sigma'] # set of parameters to optimize
        
        # directory wehere to save results
        self.r_dir = Path(r_dir)
        if not self.r_dir.exists(): self.r_dir.mkdir(parents=True)
        self.biome = biome

        # quantities to store in the output dataframe
        self.info = ['Err', 'RelErr', 'mu', 'lam', 'beta', 'sigma',
                     'mu0', 'lam0', 'beta0', 'sigma0',
                     'dEdm','dEdl','dEdb','dEds','success',
                     'h_pred', 'RelH', 'qd_pred', 'RelQd', 'q0_pred', 'RelQ0',
                     'Mass', 'RepliconNorm']

        # initialize the dataframe for results
        R=pd.DataFrame(columns=self.info)
        R.index=R.index.rename('iterations')
        self.results = R
  
    def update_results(self,theta_opt,theta0):
        
        # with the result of each opotimization
        # create a series and append it to the results df
        U=pd.Series(index=self.info,dtype=float)

        optimal_mu, optimal_lam, optimal_beta, optimal_sigma = theta_opt.x

        U['mu'],U['mu0']=optimal_mu,theta0[0]
        U['lam'],U['lam0']=optimal_lam,theta0[1]
        U['beta'],U['beta0']=optimal_beta,theta0[2]
        U['sigma'],U['sigma0']=optimal_sigma,theta0[3]
        
        U['Err']=theta_opt.fun

        U['dEdm']=theta_opt.jac[0]
        U['dEdl']=theta_opt.jac[1]
        U['dEdb']=theta_opt.jac[2]
        U['dEds']=theta_opt.jac[3]
        U['success']=int(theta_opt.success)

        theta = {'mu':optimal_mu,'sigma':optimal_sigma,'beta':optimal_beta,
                'h':self.pi['h'],'q0':self.pi['q0'],'qd':self.pi['qd'],
                'K':self.pi['K'],'lam':optimal_lam,'rho':self.pi['rho']}

        U['h_pred']=og.RSC(n=1,p=1,**theta)
        U['RelH']=np.abs(U['h_pred']/theta['h']-1) 

        U['qd_pred']=og.RSC(n=2,p=1,**theta)
        U['RelQd']=np.abs(U['qd_pred']/theta['qd']-1) 

        U['q0_pred']=og.RSC(n=1,p=2,**theta)
        U['RelQ0']=np.abs(U['q0_pred']/theta['q0']-1) 

        U['RelErr']=(U['RelH']+U['RelQd']+U['RelQ0'])/3

        U['Mass']=self.Mass()(theta_opt.x)
        U['RepliconNorm']=self.Replicon()(theta_opt.x)

        self.results = pd.concat([self.results, U.to_frame().T], ignore_index=True)
        run_file = os.path.join(self.r_dir.as_posix(),f'{self.biome}.csv')
        self.results.index=self.results.index.rename('iterations')
        self.results.to_csv(run_file,index=True)

        del U
            
    def Mass(self):
        
        h, q0, qd, K, rho = tuple(self.pi.values())
        
        def M(params):
            
            mu, lam, beta, sigma = params
            args = {'h':h,'q0':q0,'qd':qd,'sigma':sigma,'mu':mu,'beta':beta,'K':K,'lam':lam,'rho':rho}
            # <N^2> - <N>^2
            C = og.RSC(n=2,p=1,**args) - og.RSC(n=1,p=2,**args)
            return  1 - beta*(sigma*rho)**2 * C 
        
        return M

    def nMass(self):
        
        h, q0, qd, K, rho = tuple(self.pi.values())
        
        def M(params):
            
            mu, lam, beta, sigma = params
            args = {'h':h,'q0':q0,'qd':qd,'sigma':sigma,'mu':mu,'beta':beta,'K':K,'lam':lam,'rho':rho}
            # <N^2> - <N>^2
            C = og.RSC(n=2,p=1,**args) - og.RSC(n=1,p=2,**args)
            return  beta*(sigma*rho)**2 * C - 1
        
        return M
       
    def Replicon(self):
        
        h, q0, qd, K, rho = tuple(self.pi.values())
        
        def R(params):
            
            mu, lam, beta, sigma = params
            args = {'h':h,'q0':q0,'qd':qd,'sigma':sigma,'mu':mu,'beta':beta,'K':K,'lam':lam,'rho':rho}
            # <N>^4 +<N^2>^2 - 2*<N^2><N>^2
            C = og.RSC(n=1,p=4,**args) + og.RSC(n=2,p=2,**args) - 2*og.RSC(n=2,p=2,q=2,r=2,**args)
            return  1 - (beta*sigma*rho)**2 * C 
        
        return R

    def objective_function(self):
        
        h, q0, qd, K, rho = tuple(self.pi.values())
        #print(h, q0, qd, K, rho)
        def DGLVE(params):
        
            mu, lam, beta, sigma = params
            
            args = {'mu':mu,'sigma':sigma,'beta':beta,'h':h,'q0':q0,'qd':qd,'K':K,'lam':lam,'rho':rho}

            H_err = 0.5*(1-og.RSC(n=1,p=1,**args)/h)**2
            Q0_err = 0.5*(1-og.RSC(n=1,p=2,**args)/q0)**2
            Qd_err = 0.5*(1-og.RSC(n=2,p=1,**args)/qd)**2

            return H_err+Q0_err+Qd_err
        
        return DGLVE
    
    def generate_theta0(self,f_up=1e2):

        f = self.objective_function()
        
        f0 = np.nan

        while np.isnan(f0) or f0 >= f_up:
            
            '''
            s0=np.random.uniform(*self.bounds0[3])
            log_b0_min = -np.log10((self.pi['qd']-self.pi['q0'])*s0**2)

            theta0=[np.random.uniform(*self.bounds0[0]), 
                    10**(np.random.uniform(*self.bounds0[1])), 
                    10**(np.random.uniform(log_b0_min,4)), 
                    s0 ]
            '''
            s0=np.random.uniform(*self.bounds0[1])

            # sample beta such that the massive term is 1/2 (eg 1/2 max_m)
            theta0=[np.random.uniform(*self.bounds0[0]),  
                    10**(np.random.uniform(-9,-1)),
                    1/(2*(self.pi['qd']-self.pi['q0'])*s0**2),
                    s0]
            
            f0=f(theta0)

        return theta0
                     
    def exploreSeq(self):

        obf = self.objective_function()
        bounds = [(-1, 100), (1e-9, 1e-1), (1e-2, 1e6), (0, 10)]

        for t in range(self.search):
                        
            with ThreadPoolExecutor() as executor:
            
                try: 
                    theta0 = self.generate_theta0()
                    args = {'fun':obf,'x0':theta0,'bounds':bounds,'options':{'maxiter':1000}}
                    future = executor.submit(minimize,**args)
                    result = future.result(timeout=40)
                    self.update_results(theta_opt=result,theta0=theta0)

                except TimeoutError: pass

            print(t)

def main():
    
    # parse arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("-e","--environment",default='gutH',type=str,help='The environment to analyse.')
    parser.add_argument("-r","--root",default='../..',type=str,help='The project root folder.')
    args = parser.parse_args()

    # upload order parameters
    op_file = os.path.join(args.root,'data/configs',f'{args.environment}.csv')
    data = pd.read_csv(op_file,index_col='parameter')
    data = data[args.environment].to_dict()

    # set growth rate - carrying capacity proportionality constant to one
    data['rho']=1
    del data['cut']
    print(tuple(data.values()))

    # mu, log lam, log beta, sigma
    bounds0 = [(-1, 1), (0, 10)]
    
    # initialize time measurement of the code
    start_time = time.time()

    G = OPSearch(pi=data, bounds0=bounds0, r_dir=os.path.join(args.root,f'data/inference_results'), biome=args.environment )
    G.exploreSeq()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == '__main__':
    main()
