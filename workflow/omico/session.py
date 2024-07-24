import table as tb
import plot as pl
import sys

sys.path.append('../model')

#import ecomodel as em 

import pandas as pd
import numpy as np
import os
import subprocess
import json

PROJ_ROOT='../..'

colorX={('all','kaiju','RefSeq','relative_abundance'):'#587065',
        ('H','kaiju','RefSeq','corePFAM'):'#FF5500',
        ('U','kaiju','RefSeq','corePFAM'):'#5D8AA8',
        ('H','kaiju','RefSeq','relative_abundance'):'#D6C7B8',
        ('U','kaiju','RefSeq','relative_abundance'):'#BF2443',
        ('all','kraken','RefSeq','relative_abundance'):'#6A7310',
        ('H','kraken','RefSeq','relative_abundance'):'#FFD500',
        ('U','kraken','RefSeq','relative_abundance'):'#59B300'}

def check_path( target_path ):

    if os.path.exists(target_path):  pass
    
    else: 

        f=PROJ_ROOT

        for s in target_path.split('/'):

            f=os.path.join(f,s)
            if os.path.exists(f):
                print(f'{f}: folder exists')
            else: 
                print(f'mkdir {f}')
                subprocess.run(['mkdir',f])

    out_path = os.path.join(PROJ_ROOT,target_path)

    if os.path.exists(out_path): 
        print(f'<<{out_path}>>: available')
        return out_path
    else: 
        print(f'<<{out_path}>> creation: failed')

def tables_session(engine,database,cuts,protocol='relative_abundance',group='',phenotype_map=pd.Series(),nbins=0,digits=0,partition={},re_group={},PROJ_ROOT='../..'):
    
    print(f'SESSION > engine:{engine} / databse:{database} / group:{group} / protocol:{protocol}\n')
    
    # path settings
    TABLE_DIR = os.path.join(PROJ_ROOT,'data/JKR2meta/')
    METADATA_DIR = os.path.join(PROJ_ROOT,'data/metadata/aggregated')     
        
    metadata = pd.read_csv(os.path.join(METADATA_DIR,'metadata_db.csv'),index_col='run')
    
    E = ['kraken','kaiju']
    D = ['RefSeq','corePFAM','UHGG']
    
    # load the specified table
    if engine in E and database in D:
                
        print('Loading',os.path.join(TABLE_DIR,f'{engine}_{database}_table.csv') )
        
        T = pd.read_csv(os.path.join(TABLE_DIR,f'{engine}_{database}_table.csv'),index_col='taxa',sep=',')
        T = T.fillna(0)
        
        if protocol=='corePFAM':
            
            T_pfam = pd.read_csv(os.path.join(TABLE_DIR,f'{engine}_corePFAM_table.csv'),index_col='taxa',sep=',')
            T_pfam = T_pfam.fillna(0)
            
        samples = list(set(T.columns).intersection( metadata.index ))
        
        metadata = metadata.loc[ samples ]
        
        # set up the partition

        if group!='':

            if group in list(metadata.columns):

                print(f'Phenotype from metadata: {group}')

                if re_group!={}: metadata[group]=metadata[group].map( re_group )

                phenotype_map = metadata[group]

            else: 

                print(f'Custom phenotype: {group}')

            if pd.api.types.is_numeric_dtype( phenotype_map ):

                print(f'Numeric phenotype')

                phenotype_map=phenotype_map.fillna(0)
                group_bins = np.linspace( phenotype_map.min(), phenotype_map.max(), nbins+1)
                binned_group = pd.cut( phenotype_map,bins=group_bins, include_lowest=True )

                phenotype=[]
                intervals = {}

                for c in binned_group.unique():
                    
                    a =  round(0.5*( c.left+c.right),digits)
                    phenotype.append( a )
                    intervals[a]=c 

                phenotype.sort()
                partition = {}

                for p in phenotype:

                    partition[p]= list( binned_group[ binned_group==intervals[p] ].index )

            else:

                print(f'Categorical phenotype')
            
                phenotype = metadata[group].dropna().unique()
                for p in phenotype:  partition[p]= list( metadata[ metadata[group]==p].index )
                    
        else: 
            phenotype=['all']
            partition['all']=list(samples)
         
        #return partition,T
    
        tables,observables,color = {},{},{}
        
        for p in phenotype:
            
            print('\n'+f'Phenotype: {p}')
            T0 = T[ partition[p] ]
            
            tp,xp = {},{}
            
            for c in cuts:
                
                if protocol == 'relative_abundance': 
                    k='{:0.2e}'.format(c)
                    print(f'Cut-off: {k} / ',end='')
                    tp[c] = tb.table(T0,ra_cut=c)
                    
                if protocol == 'corePFAM': 
                    tp[c] = tb.table( tb.core_protocol(std_t=T0,core_t=T_pfam,core_cut=c) )
                    k='{:0.2e}'.format(c)
                    print(f'Core-PFAM cut: {k} / ',end='')
                    
                tp[c].built_in_transform(which=['relative','binary'])
        
                xp[c] = tp[c].get_observables(zipf=True,out=True)
                xp[c] = xp[c].sort_values(('zipf rank','original'))
    
            tables[p]=tp
            observables[p]=xp
            
            try:
                color[p]=colorX[(p,engine,database,protocol)]
            except KeyError:
                color[p]=pl.random_rgb()
                
        R = {'engine':engine,
             'database':database,
             'phenotype':list(phenotype),
             'group':group,
             'partition':partition,
             'protocol':protocol,
             'metadata':metadata,
             'T':tables,
             'X':observables,
             'color':color
             }
               
        return R  
'''
def model_session(data_session,cuts,replicas,DIR,configuration,specifics={},mad=None,taylor=None):
    
    MS = {p:data_session[p] for p in ['engine', 'database', 'phenotype', 'partition','protocol','group']}
    
    engine=MS['engine']
    database=MS['database']
    phenotype=MS['phenotype']
    group=MS['group']
    protocol=MS['protocol']
    print(f'SESSION > engine:{engine} / databse:{database} / group:{phenotype} / protocol:{protocol}\n')

    #SESSION_AD = f'{group}'

    SAMPLES_DIR = check_path( os.path.join(DIR,f'samples') )
    OBSERVABLES_DIR =  check_path( os.path.join(DIR,f'observables') )

    #check_path(SAMPLES_DIR)
    #check_path(OBSERVABLES_DIR)

    tables,observables = {},{}

    phenotype_parameters={} 

    for p in MS['phenotype']:

        phenotype_parameters[p]={}

        print('\n'+f'Phenotype: {p}')
        
        # get data without the cutoff
        print(cuts,data_session['T'][p].keys())
        T0 = data_session['T'][p][ cuts[0] ]
        X0 = data_session['X'][p][ cuts[0] ]
        
        # initialize model object
        Gp = em.CompoundSchlomilch(D=T0)
        
        # fit taylor
        if specifics['scaling_family']=='taylor': tau=2 
        elif specifics['scaling_family']=='poisson': tau=1
        else: fit_taylor=True

        scale=specifics['mad_scale']
        xt=(X0[f'{scale} mean']['original'].values)
        xt=np.log10(xt[xt>0])
        yt=(X0[f'{scale} var']['original'].values)
        yt=np.log10(yt[yt>0])

        if specifics['taylor']=='empirical':
            Gp.fit_taylor(fit=False,taylor={'slope':tau,'intercept':yt.mean()-tau*xt.mean()})
        elif specifics['taylor']=='fit': 
            Gp.fit_taylor(fit=fit_taylor)
        elif specifics['taylor']=='custom':
            Gp.fit_taylor(fit=False,taylor=specifics['taylor'])
        
        if specifics['mad']=='fit':
            Gp.fit_mad(scale=specifics['mad_scale'],ensemble=200,cut=-50,model=specifics['mad_model'],cut_field='loc')
        elif specifics['mad']=='empirical':
            Gp.fit_mad(fit=False) 
        elif specifics['mad']== 'custom':
            Gp.fit_mad(fit=False,mad=specifics['mad_model'])    

        tp,xp = {},{}
        
        for c in cuts:
            
            if specifics['mad']=='fit':
                Gp.sample_parameters(mode='random',samples=100000)
            elif specifics['mad']=='empirical':
                Gp.sample_parameters(mode='empirical')
            elif specifics['mad']=='custom':
                Gp.sample_parameters(mode='write',samples=mad)

            Experiments = pd.DataFrame( index= Gp.data.components )
            Experiments.index=Experiments.index.rename(Gp.data.annotation)

            xp[c]={}
            
            for r in range(replicas):
            
                print(f'replica:{r} > ',end='')
                if MS['protocol']=='corePFAM':

                    U=data_session['T'][p].form['relative'].replace(0,np.nan).values.flatten()
                    ra_c = U[~np.isnan(U)].min()

                else: ra_c=c

                Gp.sample_model(rank_conservation=True,ra_cut=ra_c )
                
                E = Gp.sample.form['original']
                E.index=E.index.rename(Gp.data.annotation)

                file = f'{c}_{p}_{r}.csv.zip'
                E.to_csv(os.path.join(SAMPLES_DIR,file),compression="gzip")
                Gp.sample_observables.to_csv(os.path.join(OBSERVABLES_DIR,file),compression="gzip")
                
            print()
            
            Experiments=Experiments.fillna(0)
                
            tp[c]=tb.table(Experiments)
                
        tables[p]=tp
        observables[p]=xp

        phenotype_parameters[p]['taylor']=Gp.taylor
        phenotype_parameters[p]['mad']=Gp.mad
    
    configuration['model_parameters']=phenotype_parameters

    json_object = json.dumps(configuration)
 
    with open(os.path.join('../..',DIR,"config.json"), "w") as outfile:
        outfile.write(json_object)

    R = {'engine':engine,
         'database':database,
         'phenotype':MS['phenotype'],
         'partition':MS['partition'],
         'protocol':MS['protocol'],
         'T':tables,
         'X':observables,
         'color':'#B284BE' }
               
    return R  
'''