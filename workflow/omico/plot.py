import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

import os
import seaborn as sns

import random

import math
from math import pi

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

plt.style.use('classic')

def standard_ax(figsize=(6,6),ticksdirection='in'):

    fig, ax = plt.subplots(figsize=figsize)

    ax.tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction=ticksdirection)
    
    for axis in ['top','bottom','left','right']:  
        ax.spines[axis].set_linewidth(3)
    
    ax.set_facecolor('#F2F2F2')
    
    fig.patch.set_facecolor('#FFFFFF')
    
    ax.grid(color='#000000',linestyle='--',alpha=0.75) 

    return fig, ax


def random_rgb():
    
    col = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    
    return col[0]

def complementary_rgb(color):
    color = color[1:]
    color = int(color, 16)
    comp_color = 0xFFFFFF ^ color
    comp_color = "#%06X" % comp_color

    return comp_color

def binning(x,y,n_bins,delta_x=0,delta_y=0,scale='linear',custom_bins=None):
    
    X = pd.DataFrame(columns=['x','y'])
    
    X['x']=x
    X['y']=y
    
    X=X.dropna()
    B=dict()
    
    x=x.values
    
    if scale=='linear':
        x_bins=np.linspace(start=x.min()-delta_x,stop=x.max()+delta_x,num=n_bins)
    if scale=='log':
        x_bins=np.logspace(start=np.log10(x).min()-delta_x,stop=np.log10(x).max()+delta_x,num=n_bins)
    if scale=='custom':
        x_bins=custom_bins
    
    label = [str(i) for i in range(n_bins-1)]
    #print( x_bins)
    X['b']=pd.cut(X['x'],x_bins,labels=label)

    B['x_mean'] = [X['x'][ X['b']==i ].mean() for i in label ]
    B['x_std']  = [X['x'][ X['b']==i ].std() for i in label ]
    
    y=y.values
    
    y_bins=np.linspace(start=y.min()-delta_y,stop=y.max()+delta_y,num=n_bins)
    y_dig = np.digitize(x=y, bins=y_bins)

    B['y_mean'] = [X['y'][ X['b']==i ].mean() for i in label ]
    B['y_std']  = [X['y'][ X['b']==i ].std() for i in label ]
    
    return B

def ecdf4plot(seq, assumeSorted = False):
    """
    In:
    seq - sorted-able object containing values
    assumeSorted - specifies whether seq is sorted or not
    Out:
    0. values of support at both points of jump discontinuities
    1. values of ECDF at both points of jump discontinuities
       ECDF's true value at a jump discontinuity is the higher one    """
    if not assumeSorted:
        seq = sorted(seq,reverse=True)
    prev = seq[0]
    n = len(seq)
    support = [prev]
    ECDF = [0.]
    for i in range(1, n):
        seqi = seq[i]
        if seqi != prev:
            preP = i/n
            support.append(prev)
            ECDF.append(preP)
            support.append(seqi)
            ECDF.append(preP)
            prev = seqi
    support.append(prev)
    ECDF.append(1.)
    return support, ECDF
    

def overwrite_plot( data, options, save=False):
    
    general = options['general']
    options.pop('general')
    
    fig, ax = plt.subplots(1,figsize=general['figsize'])
    #fig.patch.set_facecolor('white')

    
    for d in data.keys():
                
        x, y = data[d]['data']
        
        if data[d]['method']=='scatter': ax.scatter(x,y,**options[d])
        if data[d]['method']=='plot': ax.plot(x,y,**options[d])
        if data[d]['method']=='hist': ax.hist(x=x,**options[d])
    

    ax.set_xscale(general['x_scale'])
    ax.set_yscale(general['y_scale'])
    
    ax.set_xlabel(general['x_label'],fontsize=15)
    ax.set_ylabel(general['y_label'],fontsize=15)
    
    ax.legend()

    try:
        ax.set_xlim(general['x_lim'])
    except KeyError: pass

    try:
        ax.set_ylim(general['y_lim'])
    except KeyError: pass
        
    if save==True:
        fig.savefig(general['save'])
    
    return ax

def powerlaw_plot(x,y,params,title,figsize,xlim=[],ylim=[],save=False):
    
    fig, ax = plt.subplots(1, 1,figsize=figsize)
    fig.suptitle(title,style='italic')

    fig.patch.set_facecolor('white')
    
    ax.scatter(x,y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(x,np.exp(params['intercept'])*x**params['slope'],label='$ \overline{x}^{ \\nu } $',color='tab:orange',linewidth=1.5)
    
    ax.set_xlabel('Mean relative abundance, $\overline{x}  $')
    ax.set_ylabel('Variance, $\sigma_{ \overline{x} }^2$')
    
    if len(xlim)!=0 or len(ylim)!=0:
        ax.set_xlim([xlim[0],xlim[1]])
        ax.set_ylim([ylim[0],ylim[1]])
    
def block_statistics(table,sa,remove_sa,bins,xlim,ylim,xlabel,ylabel,figsize,yscale='log'):
    
    # Z will be a tuple to tuple vocabulary: for each tuple of sectors we will get the tuple of their components
    Z=dict()
    # original partition of the table
    L=table.partitions[sa]
    
    K=list(table.partitions[sa].keys())

    for i in K:
        item = L[i]
        Z[(i,i)]=(L[i],L[i])

        L = {k:L[k] for k in list(L.keys())[1:]}#L[1:]
        for j in list(L.keys()):
            Z[(i,j)]=(item,L[j])
    
    # remove eventually undesired sectors
    for s in remove_sa:
        del Z[s]

    k=0
    
    fig, ax = plt.subplots(ncols=len(Z.keys()),figsize=figsize)
    fig.patch.set_facecolor('white')

    for i in Z.keys():
        
        # slice the table
        P=table.form['original'].loc[ Z[ (i[0],i[1]) ] ]

        if P.shape[0]==P.shape[1]:
            
            mask = np.ones(P.shape,dtype='bool')
            mask[np.triu_indices(len(P))] = False

        else: mask=True

        p=P[(P!=np.nan)&mask].values.flatten()
        p=p[~np.isnan(p)]

        ax[k].tick_params(axis='both', which='major', labelsize=20,length=12.5,width=3,direction='in')
        for axis in ['top','bottom','left','right']: ax[k].spines[axis].set_linewidth(3)

        ax[k].hist(x=p,bins=bins,density=True,color='#886591', edgecolor = "black")
        ax[k].set_title(i[0]+','+i[1],fontsize=20)
        ax[k].set_xlim(xlim)
        ax[k].set_ylim(ylim)
        
        ax[k].set_yscale(yscale)

        if k>0: ax[k].set(yticklabels=[])

        k+=1
        
    ax[2].set_xlabel(xlabel,fontsize=25)
    ax[0].set_ylabel(ylabel,fontsize=25)

def block_heatmap(X,x_blocks,y_blocks,triangular=False,center=None,cbar=False,sort_x=False,sort_y=False,show_x_labels=True,show_y_labels=True,xlabel='',ylabel='',label='',save=False,cmap='rocket_r',bad_col='#FFFFFF',figsize=(8,8),IMG_FOLDER=''):

    A=X.copy()
                 
    if show_x_labels == True: xbl=list(x_blocks.keys())
    else: xbl=None
            
    if show_y_labels == True: ybl=list(y_blocks.keys())
    else: ybl=None
        
    print(ybl)
    
    B=A#1*(np.abs(A.astype(bool)))
    
    if sort_x==True: Bx = B.sum(axis=0)
    if sort_y==True: By = B.sum(axis=1)
    
    x_index = []
    
    #return Bx, By

    for b in x_blocks.keys(): 
        
        xb = x_blocks[b]
        
        if sort_x==True:
            
            xb = list(Bx[x_blocks[b]].sort_values(ascending=False).index)
            
        x_index += xb
            
    y_index = []
    
    for b in y_blocks.keys(): 
        
        yb = y_blocks[b]
            
        if sort_y==True:
            
            yb = list(By[y_blocks[b]].sort_values(ascending=False).index)
            
        
        y_index += yb
            
    A = A.loc[y_index,x_index]
     
    fig, ax = plt.subplots(1,1,figsize=figsize)
    
    #fig.patch.set_facecolor('white')
    #fig.patch.set_alpha(0)
    #sns.set_style("white")
    a_kws={"size": 20}; c_kws={'label':label, "shrink": .4};             
       
    if triangular == True:
        mask = np.triu(np.ones_like(A))
    else:
        mask = None

    cmap = matplotlib.cm.get_cmap(cmap)
    #cmap.set_bad(bad_col,alpha=1)

    ax = sns.heatmap(A, cmap=cmap, mask=mask,center=center,
                     xticklabels=xbl, yticklabels=ybl,
                     cbar=cbar,cbar_kws=c_kws,annot_kws=a_kws )

    ax.set_facecolor('#ffffff')
        
    ax.set_aspect('equal','box')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    x_l=np.zeros(len(x_blocks.keys()))
    x_s=[len(i) for i in x_blocks.values()]
    x_l[0]=x_s[0]
    for i in range(2,len(x_s)+1): x_l[i-1]=x_l[i-2]+x_s[i-1]
    ax.vlines(x_l[:-1], color=bad_col,*ax.get_ylim(),linewidth=2.)
    
    l=np.array([0]+list(x_l)[:-1])
    x_ticks=l + np.array(x_s)*0.5
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=0,fontsize=20)

    y_l=np.zeros(len(y_blocks.keys()))
    y_s=[len(i) for i in y_blocks.values()]
    y_l[0]=y_s[0]
    for i in range(2,len(y_s)+1): y_l[i-1]=y_l[i-2]+y_s[i-1]
    ax.hlines(y_l[:-1], color=bad_col,*ax.get_xlim(),linewidth=2.)
    
    l=np.array([0]+list(y_l)[:-1])
    y_ticks=l + np.array(y_s)*0.5
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=20)
    
        
    if save==True:
        fig.savefig(os.path.join(IMG_FOLDER,'blockmap.png'), transparent=True, dpi=150,bbox_inches='tight' )

def matrix_statistics(Q,n_bins,title,no_zero=False,eigenvalues=False,save=False,label=[],logscale=[False,False,False,False]):
    
    QS=Q.shape[0]
    
    if no_zero==True: 
        Qrr=Q[Q!=0].flatten()
    else: 
        Qrr=Q.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
    fig.patch.set_facecolor('white')
    
    if logscale[0]==True: 
        ax1.set_xscale('log')
        bins = np.logspace(np.log10(Qrr.min()), np.log10(Qrr.max()), n_bins)
    else:
        ax1.set_xscale('linear')
        bins= np.linspace(Qrr.min(),Qrr.max(),n_bins)
        
    if logscale[1]==True: 
        ax1.set_yscale('log')  
        
    ax1.set_xlabel(label[0]); ax1.set_ylabel('Probability density')

    ax1.hist(Qrr,density=True,bins=bins,alpha=0.5,color='#FF7E00')
    
    if eigenvalues==True:
        
        q_eig, q_eigen_vectors = np.linalg.eig(Q)
        q_eig=q_eig[ np.where(q_eig>0) ]
        
        bins_log10 = np.logspace(np.log10(q_eig.min()), np.log10(q_eig.max()), n_bins)
        
        ax2.set_xlabel(label[1])
        
        if logscale[2]==True:
            ax2.set_xscale('log')
            bins = np.logspace(np.log10(q_eig.min()), np.log10(q_eig.max()), n_bins)
        else:
            ax2.set_xscale('linear')
            bins = np.linspace(q_eig.min(),q_eig.max(),n_bins)
        
        if logscale[3]==True: 
            ax2.set_yscale('log')
            
        ax2.hist(q_eig, bins=bins, alpha=0.5, label='histogram',color='#AF002A',density=True)
    
        fig.suptitle(title)
    
        if save==True:
            fig.savefig(os.path.join(IMG_FOLDER,'corr_stat.png'))
    
def pdf_plot(samples,model,N_bins,graphic_options,figsize=(8,8),logscale=[False,False],save=False,**params):
     
    general = graphic_options['general']
    
    fig, ax = plt.subplots(1, figsize=general['figsize'], gridspec_kw={'wspace':0.2})
    fig.patch.set_facecolor('white')
    
    if logscale[0]==True:
        ax.set_xscale('log')
        bins = np.logspace(np.log10(samples.min()), np.log10(samples.max()), N_bins)
    else:
        ax.set_xscale('linear')
        bins = np.linspace(samples.min(),samples.max(),N_bins)
        
    if logscale[1]==True:
        ax.set_yscale('log')
        
    counts, bin_edges, ignored = ax.hist(x=samples, bins=bins,density=True,**graphic_options['hist'])
    
    bins_cntr =  0.5*(bin_edges[1:] + bin_edges[:-1])
    area_hist = (bins_cntr * counts).sum()
    
    samples_fit = model.pdf(bins_cntr, *list(params.values()) )

    ax.plot(bins_cntr, samples_fit * area_hist, **graphic_options['line'])

    ax.set_xlabel(general['xlabel'],fontsize=16)
    ax.set_ylabel(general['ylabel'],fontsize=16)
    ax.set_title( general['name'], fontsize = 20 )
    
    if save==True:
        fig.savefig(os.path.join(general['IMG_FOLDER']+general['name']+'.png'))
        
    return ax
    

def overwrite_pdf( data, options,density=True, n_bins=20, save=False):
    
    general = options['general']
    options.pop('general')
    
    fig, ax = plt.subplots(1,figsize=general['figsize'])
    fig.patch.set_facecolor('white')
    
    minimum, maximum = np.inf, -np.inf
    
    for d in data.keys():
        X=data[d]['data']
        minimum=min(minimum,*X)
        maximum=max(maximum,*X)
        
    print(minimum,maximum)
    if general['x_scale']=='log': 
        bins = np.logspace(np.log10(minimum), np.log10(maximum), n_bins)
    else:
        bins= np.linspace( minimum, maximum, n_bins)

    for d in data.keys():
        ax.hist(data[d]['data'],bins=bins,density=density,**options[d])


    ax.set_xscale(general['x_scale'])
    ax.set_yscale(general['y_scale'])
    
    ax.set_xlabel(general['x_label'])
    ax.set_ylabel(general['y_label'])
    
    ax.legend()
        
    if save==True:
        fig.savefig(general['save'])
    
    return ax

def density_scatter( x , y, ax = None, xlabel='',ylabel='',x_scale='log', points_label='',y_scale='linear',density_scale='linear',sort = True, marginal_hist_x=False, marginal_hist_y = False, bins = 20, figsize=(7,7), colorbar=False,**kwargs )   :
    """
    Scatter plot colored by 2d histogram
    # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
    """
    
    if ax is None :
        
        if marginal_hist_x == True or marginal_hist_y == True:
            
            fig = plt.figure(figsize=(8, 8))

            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing = 0.005

            rect_scatter = [left, bottom, width, height]

            ax = fig.add_axes(rect_scatter)

            if marginal_hist_x == True:
                rect_histx = [left, bottom + height + spacing, width, 0.2]
                ax_histx = fig.add_axes(rect_histx, sharex=ax)

            if marginal_hist_y == True:   
                rect_histy = [left + width + spacing, bottom, 0.2, height]
                ax_histy = fig.add_axes(rect_histy, sharey=ax)
            
        else:
            fig , ax = plt.subplots(figsize=figsize)
        
    if x_scale == 'log':
        bx=np.logspace(np.log10(x.min()),np.log10(x.max()),bins)
        ax.set_xscale('log')
        #x=np.log(x)
    else:
        bx=np.linspace(x.min(),x.max(),bins)
        ax.set_xscale('linear')
        #x=x
        
    if y_scale == 'log':
        by=np.logspace(np.log10(y.min()),np.log10(y.max()),bins)
        ax.set_yscale('log')
        #y=np.log(y)
    else:
        by=np.linspace(y.min(),y.max(),bins)
        ax.set_yscale('linear')
        #y=y
    
    fig.patch.set_facecolor('white')
    
    ax.set_xlabel(xlabel,fontsize = 16)
    ax.set_ylabel(ylabel,fontsize = 16)
    
    # density section
        
    data , x_e, y_e = np.histogram2d( x, y, bins = [bx,by], density = True )

    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , 
                 data , 
                 np.vstack([x,y]).T , 
                 method = "splinef2d", 
                 bounds_error = False)
    
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z,label=points_label,**kwargs)
    
    #ax.set_xlim([x.min(),x.max()])
    #ax.set_ylim([y.min(),y.max()])
    if colorbar==True:
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')
        
    if marginal_hist_x == True or marginal_hist_y == True:
            
        if marginal_hist_x == True:
            rect_histx = [left, bottom + height + spacing, width, 0.2]
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histx.hist(x, bins=bins,color='#003CB3',alpha=0.5)
            ax_histx.set_ylabel('Samples')
            ax_histx.set_ylim([0,350])
            #ax_histx.set_ytick(ax_histx.get_xtick())
            
        if marginal_hist_y == True:   
            rect_histy = [left + width + spacing, bottom, 0.2, height]
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histy.hist(y, bins=bins,orientation='horizontal',color='#003CB3',alpha=0.5)
            ax_histy.set_xlabel('Samples')
            ax_histy.set_xlim([0,350])
            ax_histy.locator_params(axis="x", nbins=4)
            #ax_histy.set_xticklabels(ax_histx.get_yticklabels())

    return fig, ax
