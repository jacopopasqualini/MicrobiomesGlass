import pandas as pd
import numpy as np
import os

closing_message = ' < < < \t < < < '

def binary_transform(X): return X.astype(bool)*1
def relative_transform(X): return X.apply(lambda x: x/x.sum(), axis=0)
def z_transform(X): return X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
def mean_transform(X): return X.apply(lambda x: x/x.mean(), axis=0)

###################################################################
# UTILITIES

def built_in_transform():
    
    bits = {}

    bits['binary'] = binary_transform
    bits['relative'] = relative_transform
    bits['z'] = z_transform
    bits['mean'] = mean_transform
    
    return bits

def empirical_distribution(X,scale='linear',n_bins=10):
    
    ''''''
    x_l = X.min()
    x_u = X.max()
    
    if scale=='linear':
        bins=np.linspace(x_l,x_u,n_bins)
    if scale=='log':
        bins=np.logspace(np.log10(x_l),np.log10(x_u),n_bins)
    
    p=np.histogram(a=X,bins=bins)
    
    bpoint = p[1]
    probs = p[0]/p[0].sum()
        
    return bpoint, probs

def size_cutoff(X,l,u):

    Y=X.copy()

     # keep only realizations that lay in the range
    S=Y.sum(axis=0)
    S=S[S>l][S<u]
    Y =  Y[S.index]

    #throw away componets that do not appear in any realization
    K=Y.sum(axis=1)
    K=K[K>0]
    Y =  Y.loc[K.index]

    del S
    del K

    return Y

def sparsity_cutoff(X,pc=1,om=1):

    Y=X.copy()

    M=(1*Y.astype(bool)).sum(axis=1)
    Y = Y.loc[ M[M>om].index ]

    B=(1*Y.astype(bool)).sum(axis=0)
    Y=Y[ B[B>pc].index ]

    del B
    del M
    
    return Y

def core_protocol(std_t,core_t,core_cut,index_name='taxon_name'):
    
    configurations = set(std_t.columns).intersection(core_t.columns)
    components     = set(std_t.index).intersection(core_t.index)
    
    std_t  = std_t[configurations].loc[components]
    core_t = core_t[configurations].loc[components]
    
    C = pd.DataFrame(columns=configurations,index=components)
    
    for s in list(configurations):
    
        r,p = std_t[s],core_t[s]
        C[s] = r.loc[ p[ p>core_cut ].index ]
    
    C=C.fillna(0)
    
    V=C.sum(axis=0)
    C=C[ V[V>0].index ]

    U=C.sum(axis=1)
    C=C.loc[ U[U>0].index ]

    C.index=C.index.rename(index_name)

    return C

###################################################################
# TABLE OBJECT

class table:
    
    def __init__(self,T,cut=True,lower_size=0,upper_size=np.inf,pc=1,om=1,ra_cut=None,verbose=False):

        if verbose==False:
            self.vm = 0
        elif verbose==True:
            self.vm = 1

        print(self.vm*' > > > table initialization: ',end='')
        
        self.form = {}

        if ra_cut!=None:

            tab=T.copy()  
            B=tab/tab.sum(axis=0)
            tab[B<ra_cut]=0
            T=tab
        
        ''' first original for external use, second for internal use and cool coding'''
        
        if pc!=1 or om!=1:
            Y = sparsity_cutoff(X=T,pc=pc,om=om)
        else:
            Y = T.copy()
            
        if cut==True:
            Z = size_cutoff(X=Y,l=lower_size,u=upper_size)
        else:
            Z = Y.copy()

        self.form['original'] = Z.copy()
        self.original = Z.copy()
        
        # name of the set of variables
        self.annotation = self.original.index.name
        self.shape = self.original.shape
        
        self.samples = list(self.original.columns)
        # name of each variable
        self.components = list(self.original.index)
        
        self.realization_size = self.original.sum(axis=0)
        #self.n_components = (self.original.astype(bool)*1).sum(axis=0)
        self.bits = built_in_transform()
        self.binned = False
        
        self.partitions = {'original':{'original':list(self.samples)}}  
        self.observables = {}

        print(self.vm*'done',end=self.vm*'\n')
        print(self.vm*closing_message,end=self.vm*'\n')
    
    def add_partition(self,partition,name):
        
        ''' a partition of the dataset is a vocabulary which has groups labels as keys
        and a list of samples as values. This will allow the observables calculation.
        To effectively add a partition to the table object, apart from the vocabulary itself,
        a name for the partition needs to be provided'''
        
        self.partitions[name]=partition
    
    def del_partition(self,name):
        
        self.partitions.pop(name)

        if name=='size':
            self.binned = False
        
    def size_partitioning(self,scale='log',n_bins=10):
        
        print(self.vm*' > > > size partitioning  \n')

        if self.binned == True:
            pass
        else:
            print(self.vm*' > > > initialization: ',end='')
            A = self.original
            #size=A.sum(axis=0) # self.size, nuovo attributo
            print('done')

            # binned samples and frequencies
            print(self.vm*' > > > size distribution: ',end='')

            bp,p_bs = empirical_distribution(X=self.realization_size,scale=scale,n_bins=n_bins)
            self.binned_sizes = bp

            bs=(.5*(bp[1:]+bp[:-1])).astype(int)
            self.size_distribution = dict(zip(bs,p_bs)) # 

            print('done')

            print(' > > > size partitioning: ',end='')
            
            # initialize sample container
            mc = {}
            for b in bs: mc[b] = []
                
             # create the structure to feed into pandas multi-index
            for t in self.realization_size.index:
                v = np.argmin(np.abs( self.realization_size[t]-bs ))
                #mc[t]=bs[v]
                mc[ bs[v] ].append(t)
                
            self.partitions['size']=mc
            print('added')
            self.binned = True
            
            print(closing_message)
        
    def built_in_transform(self,which=[]): 
        
        ''' apply the built in transforms to the dataset'''
        print(self.vm*' > > > built-in transform',end=self.vm*'\n')

        if which==[]: transforms = self.bits.keys()
        else: transforms = which
            
        for t in transforms:
            
            print(self.vm*f' > > > {t} transform: ',end=self.vm*'\n')
            f=self.bits[t]
            self.form[t]=f(self.original)
            print(self.vm*'done',end=self.vm*'\n')
        
        print(self.vm*closing_message,end=self.vm*'\n')
        
    # aggiungi i metodi remove per ogni add
    def add_transform(self,fury):
        
        ''' fury: a dictionary with function name as keys and the function itself as value'''
        for f in fury.keys():
            if f not in self.bits.keys():
                g=fury[f]
            else:
                print(f'{f}  already exists. Built-in transform will be provided')
                g=self.bits[f]
                
            print(f' > > > {f} transform: ',end='')
            self.form[f]=g(self.original)
            print('done')
            
        print(closing_message)
            
    def del_transform(self,transform):

        try:
            self.form.pop(transform)
        except KeyError:
            print(f' * * > {transform} related observables do not exist')
            pass
            
    def get_observables(self,zipf=False,out=False,grouping='original',axis=1):
        
        print(self.vm*f' > > > observables {grouping}',end=self.vm*'\n')

        #if self.observables[grouping] !=
        fake_cols = pd.MultiIndex.from_tuples( list(zip(*[['fake1'],['fake2']])))
        report = pd.DataFrame(index=self.components,columns=fake_cols)
        report.index=report.index.rename(self.annotation)                                                
        # il binsize lo devo fare a prescindere. faccio il report già binnato
        # e quello non binnato è semplicemente la media sui bin
        P = self.partitions[grouping]
        
        for f in self.form.keys():
            
            print(self.vm*' > > > {f} processing: ',end=self.vm*'\n')
            
            # create here multiindex
            # Ho le osservabili sopra e le varie partizioni sotto nel multiindex
            r = pd.DataFrame(index=self.components,columns=P.keys())
            r.index = r.index.rename(self.annotation)
            q = pd.DataFrame(index=self.components,columns=P.keys())
            q.index = q.index.rename(self.annotation)

            for p in P.keys():
                
                samples = P[p]
                
                r[p] = self.form[f][samples].mean(axis=axis)
                q[p]  = self.form[f][samples].var(axis=axis)
            
            # indice per le partizioni
            J=list(P.keys())
            # indice per le osservabili,sta sopra
            Im=[f'{f} mean']*len(J)
            Iv=[f'{f} var']*len(J)
            
            r.columns = pd.MultiIndex.from_tuples( list(zip(*[Im,J])) )
            q.columns = pd.MultiIndex.from_tuples( list(zip(*[Iv,J])) )
            
            report = report.merge(r,on=self.annotation,how='outer')
            report = report.merge(q,on=self.annotation,how='outer')
            
            print(self.vm*'done',end=self.vm*'\n')
        
        if zipf==True:

            print(self.vm*f' > > > zipf processing: ',end=self.vm*'\n')

            r = pd.DataFrame(index=self.components,columns=P.keys())
            r.index = r.index.rename(self.annotation)

            for p in P.keys():
                
                z = report['relative mean'][p].sort_values(ascending=False)
                rank = np.arange(0, z.shape[0] )+1
                r[p] = pd.Series(index=z.index,data=rank)
                
            J=list(P.keys())
            Iz=['zipf rank']*len(J)
            r.columns = pd.MultiIndex.from_tuples( list(zip(*[Iz,J])) )
            
            report = report.merge(r,on=self.annotation,how='outer')
            
            print(self.vm*'done',end=self.vm*'\n')
        
        del report['fake1']
        self.observables[grouping] = report 
        
        print(self.vm*closing_message,end=self.vm*'\n')
        
        if out==True: return report
        else: pass
        
    def del_observables(self,partition):
        try:
            self.observables.pop(partition)
        except KeyError:
            print(f' * * > {partition} related observables do not exist')
                
    def get_annotation(self):
        return self.annotation
    
    def get_shape(self):
        return self.shape
    
    def get_samples(self):
        return self.samples
    
    def get_components(self):
        return self.components