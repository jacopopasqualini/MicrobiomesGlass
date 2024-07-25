import pandas as pd
import os

top=30
data_dir = '../../data'
##########################################################
### healthy

H=pd.read_csv(os.path.join(data_dir,'inference_results','gutH.csv'),index_col='iterations')
H=H.sort_values('RelErr')
H=H.iloc[:top]

H['rank']=range(1,H.shape[0]+1)
H['env']='gutH'

H_params = pd.read_csv(os.path.join(data_dir,'configs','gutH.csv'),index_col='parameter')
H['K']=H_params.loc['K']

##########################################################
### diseased

U=pd.read_csv(os.path.join(data_dir,'inference_results','gutU.csv'),index_col='iterations')
U=U.sort_values('RelErr')
U=U.iloc[:top]
U['rank']=range(1,H.shape[0]+1)
U['env']='gutU'

U_params = pd.read_csv(os.path.join(data_dir,'configs','gutU.csv'),index_col='parameter')
U['K']=U_params.loc['K']

##########################################################
### concatenate results

X=pd.concat([H,U])
X.to_csv(os.path.join(data_dir,'inference_results','nice.csv'),index=True)