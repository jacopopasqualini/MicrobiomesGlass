import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.special import erfinv, erf
from scipy.special import gamma, hyp1f1, polygamma
from scipy.integrate import quad
from scipy.optimize import approx_fprime

import os
import sys

def dhyp1f1_da(a, b, z, epsilon=1e-6):
    func = lambda x: hyp1f1(x, b, z)
    gradient = approx_fprime(a, func, epsilon)
    return gradient

# numerical check with mathematica DONE!

# -------------------------------------------------------------
# --- expectation values over the boltzmann enseble average ---

def BoltzmannAvgN(a,b,c,n):
    
    # expectation value of abundance according to boltzmann measure with RS-Hamiltonian
    # <N^n>_{RS} = Integrate[N^(a-1+n) Exp[-(b N^2/2 - c N ) ], {N, 0, Infinity}]
    # normalized by [ Integrate[ N^(a-1) Exp[-(b N^2/2 - c N ) ], {N, 0, Infinity}]]
    f = (2/b)**(0.5*n)

    n1 = np.sqrt(b) * gamma( 0.5*(a+n) ) * hyp1f1( 0.5*(a+n), 0.5, c**2/(2*b) )
    n2 = np.sqrt(2) * c * gamma( 0.5*(a+n+1) ) * hyp1f1( 0.5*(a+n+1), 1.5, c**2/(2*b) )

    d1 = np.sqrt(b) * gamma( 0.5*(a) ) * hyp1f1( 0.5*(a), 0.5, c**2/(2*b) )
    d2 = np.sqrt(2) * c * gamma( 0.5*(a+1) ) * hyp1f1( 0.5*(a+1), 1.5, c**2/(2*b) )

    return f*(n1+n2)/(d1+d2)

def BoltzmannAvgLog(a,b,c,n):

    # expectation value of abuundance with logs
    # <N^nLogN> = [ Integrate[ Log[N] N^(a + n - 1) Exp[-(b N^2/2 - c N ) ], {N, 0, Infinity}]]
    # normalized by [ Integrate[ N^(a - 1) Exp[-(b N^2/2 - c N ) ], {N, 0, Infinity}]]
    
    f=(2/b)**(n/2) /np.sqrt(8)
    
    n1= np.sqrt(2*b)*gamma((a+n)/2)
    n2= (np.log(2/b)+polygamma(0,(a+n)/2))*hyp1f1((a+n)/2,1/2,c**2/(2*b))+dhyp1f1_da((a+n)/2,1/2,c**2/(2*b))

    n3= 2*c*gamma((1+a+n)/2)
    n4= (np.log(2/b)+polygamma(0,(1+a+n)/2))*hyp1f1((1+a+n)/2,3/2,c**2/(2*b))+dhyp1f1_da((1+a+n)/2,3/2,c**2/(2*b))

    d1 = np.sqrt(b) * gamma( a/2 ) * hyp1f1( 1/2*(a), 1/2, c**2/(2*b) )
    d2 = np.sqrt(2) * c * gamma( (a+1)/2 ) * hyp1f1((a+1)/2, 3/2, c**2/(2*b) )
    
    return f*(n1*n2+n3*n4)/(d1+d2)

# --------------------------------------
# --- external average over disorder ---

def RSC(h,q0,qd,sigma,mu,beta,K,lam,rho,n,p,q=0,r=0):
    
    '''
    replica-symmetric correlator
    \overline{ <N^n>^p <N^q>^r }
    '''
    
    # Define the integrand function
    def DisorderedAvg(z):
        
        a = beta * lam
        b = beta*(rho - beta * (qd-q0) * (rho*sigma)**2)
        c = beta * rho * ( K - mu*h + z*np.sqrt(q0)*sigma )
        mass=(rho - beta * (qd-q0) * (rho*sigma)**2)
        
        #print('Mass=',mass,'qd',qd,'q0',q0,'sigma',sigma,'beta',beta)
        Dz = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
        
        #print('RSC:',a,b,c)
        Nn = BoltzmannAvgN(a=a,b=b,c=c,n=n)
        
        if r!=0: Nq = BoltzmannAvgN(a=a,b=b,c=c,n=q)
        else: Nq=1
            
        return Dz * (Nn)**p * (Nq)**r

    # Perform the numerical integration
    result, _ = quad(DisorderedAvg, -10, 10, epsabs=1e-10, epsrel=1e-10)

    return result

def LogRSC(h,q0,qd,sigma,mu,beta,K,lam,rho,n=0,q=1,r=0):
    
    '''
    replica-symmetric correlator
    \overline{ <N^n * Log(N)> <N^q>^r }
    '''
    
    # Define the integrand function
    def DisorderedAvg(z):
        
        a = beta * lam
        b = beta*(rho - beta * (qd-q0) * (rho*sigma)**2)
        c = beta * rho * ( K - mu*h + z*np.sqrt(q0)*sigma )

        Dz = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
        
        zl = BoltzmannAvgLog(a=a,b=b,c=c,n=n)
        
        if r!=0: zn = BoltzmannAvgN(a=a,b=b,c=c,n=q)
        else: zn=1
            
        return Dz * zn * zl

    # Perform the numerical integration
    result, _ = quad(DisorderedAvg, -10, 10, epsabs=1e-10, epsrel=1e-10)

    return result

def replicon(h,q0,qd,sigma,mu,beta,K,lam,rho):

    theta = {'h':h,'q0':q0,'qd':qd,'sigma':sigma,'mu':mu,'beta':beta,'K':K,'lam':lam,'rho':rho}
    # <N>^4 +<N^2>^2 - 2*<N^2><N>^2
    C = RSC(n=1,p=4,**theta) + RSC(n=2,p=2,**theta) - 2*RSC(n=2,p=1,q=1,r=2,**theta)
    return (beta*sigma*rho)**2 * ( 1 - (beta*sigma*rho)**2 * C )

def marginal_sigma_replicon(h,q0,qd,mu,beta,K,lam,rho):
    
    def condition(sigma):
        N4     = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=1,p=4) 
        N_22   = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=2,p=2)
        N_2222 = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=2,p=1,q=1,r=2)

        mar = 1 - (beta*sigma*rho) **2 * ( N4 + N_22 - 2 * N_2222 )
        return mar**2

    return condition

def marginal_beta_replicon(h,q0,qd,mu,sigma,K,lam,rho):
    
    def condition(beta):
        N4     = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=1,p=4) 
        N_22   = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=2,p=2)
        N_2222 = RSC(h=h,q0=q0,qd=qd,sigma=sigma,mu=mu,beta=beta,K=K,lam=lam,rho=rho,n=2,p=1,q=1,r=2)

        mar = 1 - (beta*sigma*rho) **2 * ( N4 + N_22 - 2 * N_2222 )
        return mar**2

    return condition

