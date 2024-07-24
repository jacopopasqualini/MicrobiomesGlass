import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import opgd as og

from scipy import special as sc

from scipy.special import erfinv, erf
from scipy.special import gamma
from scipy.special import hyp1f1

from scipy.special import gamma, hyp1f1, polygamma
from scipy.integrate import quad
from scipy.optimize import approx_fprime

import concurrent.futures
import functools

import sys

import opgd as og

sys.path.append('./omico')

import plot as pl
import table as tb
import session as es

######################################
### get order parameters from data

def GetOPs(X,cut=-np.inf):

    h  = (np.ma.array(X,mask= X<cut ).mean(axis=0)).mean()
    qd = (np.ma.array(X**2,mask= X<cut ).mean(axis=0)).mean()
    q0 = (np.ma.array(X,mask= X<cut ).mean(axis=0)**2).mean()
    K  = (np.ma.array(X,mask= X<cut ).max(axis=0)).mean()

    return h,qd,q0,K

######################################
### ada cavity pdf sampler

class CavityDGLV():
    
    """
    PDF and samples for the DGLV cavity distribution.
    The PDF is:
        p(N) \propto N^(a-1) * exp( -b*N^2/2 + c N )
        a = beta*lambda, b=beta*(1-(qd-q0)*sigma^2), c=beta*zeta
    """
    
    def __init__(self,a,b,c):
        
        # set the partition function of the theory
        f = 2**(0.5*(a-1)) / b**(0.5*(a+1))
        Z1 = np.sqrt(0.5*b) * gamma(0.5*a) * hyp1f1(0.5*a, 0.5, 0.5*c**2/b)
        Z2 = c * gamma( 0.5*(1+a) ) * hyp1f1(0.5*(1+a),1.5, 0.5*c**2/b )
    
        self.Z =  f*(Z1+Z2)
        
        # save theory parameters
        self.a=a; self.b=b; self.c=c

    def pdf(self):
        
        # return the probability density function of the theory
        def P(N):
            
            return N**(self.a-1)*np.exp( -0.5*self.b*N**2 + self.c*N ) / self.Z
        
        return P

    def plot_pdf(self,title,PLT_DIR='../plots/SCEs/cavityPDF'):

        #print('a:',self.a,'b:',self.b,'c:',self.c)
        
        x=np.logspace(np.log10(self.Nmin),self.log10(self.Nmax),1000)
        P=self.pdf()
        fig,ax=plt.subplots(figsize=(15,7))

        ax.plot(x,P(x),linewidth=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Species abundance, N')
        ax.set_ylabel('pdf(N)')

        fig.savefig(os.path.join(PLT_DIR,f'{title}.png'), dpi=150,bbox_inches='tight')

    def rvs(self,S,R=2000,z_divergence=True,e1=0.01,e2=0.01,Tmax=2000,Nmin=1e-17,Nmax=10):
        
        # S: number of samples
        # N_max: maximum value reached by integration
        # R: resolution of the piecewise approximation
        # Tmax: maximum number of samples to generate in order to match analytical prediciton

                # save numerical integration bounds
        avgN = og.BoltzmannAvgN(a=self.a,b=self.b,c=self.c,n=1)
        avgN2 = og.BoltzmannAvgN(a=self.a,b=self.b,c=self.c,n=2)
        
        # this trick helps in sampling rare species
        if z_divergence:
            x = np.logspace(np.log10(Nmin),np.log10(Nmax),R)
        else:
            np.linspace(Nmin, Nmax, R)

        # get point-wise estimates for the cumulative density function
        cdf_values = [quad( self.pdf(),Nmin, xi)[0] for xi in x]
        cdf_values = np.array( cdf_values )
        cdf_values /= cdf_values[-1]

        # intial values of errors
        Err1,Err2=1,1
        # try many samples such that the generated ensamble circmveits numerical integration problem instability

        for t in range(Tmax):
            # generate random values
            r = np.random.uniform(0, 1, size=S)

            # approximate the cdf as a piecewise linear function.
            # as such can always be locally inverted
            N = np.interp(x=r, xp=cdf_values, fp=x)

            Err1 = np.abs( 1-np.mean(N)/avgN )
            Err2 = np.abs( 1-np.mean(N**2)/avgN2 )
            
            if Err1<e1 and Err2<e2: 
                break
        
        return N

def TableSampler(D,h,q0,qd,sigma,mu,beta,K,lam,rho,index_name='taxa'):
    
    # input: thet set of a,b,c parameters of the dglv sad. 
    #        a pandas dataframe 

    # prepare dataframe
    X=pd.DataFrame(columns=D.columns,index=D.index)
    S=X.shape[0]

    a=beta*lam
    b=beta*(rho-beta*(qd-q0)*(rho*sigma)**2)
    
    for r in X.columns:
        
        # generate the disorder
        z = np.random.normal(loc=0,scale=1)
        c=beta*( K - mu*h + np.sqrt(q0)*sigma*z )

        # prepare SAD sampler
        C = CavityDGLV( a=a,b=b,c=c )
        
        # sample bare SAD abundances, normalize and sample with multinomial

        X[r]=pd.Series( data=C.rvs(S=S), index=X.index )
        
    X.index.name = index_name
        
    return X

def SAD(a,b,c,S,bins=2000):

    C = CavityDGLV(a=a, b=b, c=c)

    return C.rvs(S=S,R=bins)

def get_theta(mu,sigma,K,h,q0,qd,beta,lam,S,rho=1):

    a=beta*lam
    b=beta*(rho-beta*(qd-q0)*(rho*sigma)**2)
    c=beta*(K - mu*h + np.sqrt(q0)*sigma*np.random.normal(0,1))

    return a,b,c,S

def TableSamplerParallel(D, h, q0, qd, sigma, mu, beta, K, lam, rho, sampling_bins, index_name='taxa'):

    X = pd.DataFrame(columns=D.columns, index=D.index)
    S, R = X.shape[0], X.shape[1]

    p = {'h':h,'qd':qd,'q0':q0,'sigma':sigma,'mu':mu,'beta':beta,'K':K,'lam':lam,'rho':rho,'S':S}

    Theta = [ get_theta(**p) for _ in range(R) ]

    for i in range(len/Theta): Theta[i]['bins']=sampling_bins

    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures = [executor.submit(SAD, *args) for args in Theta]

    for f,r in zip(concurrent.futures.as_completed(futures),X.columns):

        X[r] = pd.Series(index=X.index,data=f.result())
    
    X.index.name = index_name

    return X


def DOC(T):
    
    couples = list(combinations(T.columns, 2))
    couples = [f'{c[0]}/{c[1]}' for c in couples]

    DOC=pd.DataFrame(index=couples,columns=['dissimilarity','overlap'],dtype=float)

    #print(DOC)
    for c in DOC.index:
        
        sample = c.split('/')
        Y,Z=X[sample[0]],X[sample[1]]
        
        # get only shared species
        s=list(set(Z[Z>0].index).intersection(Y[Y>0].index))
        Y,Z=Y.loc[s],Z.loc[s]

        # evaluate overlap
        DOC.loc[c,'overlap'] = 0.5*(Y+Z).sum()

        # re-normalize
        Y/=Y.sum()
        Z/=Z.sum()
        
        m=0.5*(Z+Y)
        DOC.loc[c,'dissimilarity']=np.sqrt(0.5*(sc.rel_entr(Z,m).sum()+sc.rel_entr(Y,m).sum()))
        
    return DOC
