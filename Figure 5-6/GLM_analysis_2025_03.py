# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:00:03 2025

@author: Jong Hoon Lee
"""

# import packages 

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet,ElasticNetCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns
from os.path import join as pjoin
from numba import jit, cuda

from scipy.stats import zscore

# %% File name and directory

# change fname for filename
# fname = 'CaData_all_all_session_v2_corrected.mat'
fname = 'CaData_all_session_v3_corrected.mat'

fdir = 'D:\Python\Data'


# %% Helper functions for loading and selecting data
np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 

def find_good_data_Ca(t_period):
    D_ppc = load_matfile_Ca()
    good_list = []
    t_period = t_period+prestim

    for n in range(np.size(D_ppc,0)):
        N_trial = np.size(D_ppc[n,2],0)
    
        ttr = D_ppc[n,4][0][0]

    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y[:200,:]) > 0.5 :
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[200:ttr+26,:]) > 0.5:
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[ttr+26:N_trial,:])> 0.5 :
            good_list = np.concatenate((good_list,[n]))
    
    return good_list


# %% import data helper functions

def import_data_w_Ca(D_ppc,n,window,c_ind):    
    # For each neuron, get Y, neural data and X task variables.  
    # Stim onset is defined by stim onset time
    # Reward is defined by first lick during reward presentation
    # Lick onset, offset are defined by lick times
    # Hit vs FA are defined by trial conditions
    
    

    N_trial = np.size(D_ppc[n,2],0)

    ### Extract Ca trace ###
    Yraw = {}
    Yraw = D_ppc[n,0]
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    
    
    Y = np.zeros((1,len(time_ind)-1))   
    for t in np.arange(len(time_ind)-1):
        Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
    
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    L_all_onset = np.zeros((1,len(time_ind)-1))
    L_all_offset = np.zeros((1,len(time_ind)-1))
    # Rt = np.zeros((1,len(time_ind)-1))
    Ln = np.array(D_ppc[n,1])
    InterL = Ln[1:,:]- Ln[:-1,:]
    lick_onset= np.where(InterL[:,0]>2)[0] # lick bout boundary =2
    lick_onset = lick_onset+1
    lick_offset = lick_onset-1
    
    for l in np.floor(D_ppc[n,1]*(1e3/window)): 
        L_all[0,int(l[0])-1] = 1 
            
    for l in np.floor(Ln[lick_onset,0]*(1e3/window)):
        L_all_onset[0,int(l)-1] = 1
    
    for l in np.floor(Ln[lick_offset,0]*(1e3/window)):
        L_all_offset[0,int(l)-1] = 1 
    
    ### Extract Lick End ###
    

    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)

    X = D_ppc[n,2][:,2:6] # task variables
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    
    
    ### Create variables ###
    ED1 = 5 # 500ms pre, 1second post lag
    ED2 = 10
    stim_dur = 5 # 500ms stim duration
    delay = 10 # 1 second delay
    r_dur = 5 # 2 second reward duration (was 10) 
    ED3 = 30 # 4 seconds post reward lag
    ED4 = 70
    ED5 = 50
    ED_hist1 = 50 # 4 seconds pre-stim next trial
    ED_hist2 = 15 # 1.5 seconds post-stim next trial
    h_dur = 5
    
    X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(Y,1)))
    X3_Lick_offset = np.zeros_like(X3_Lick_onset)
    
    X3_Lick_onset[0,:] = L_all_onset
    X3_Lick_offset[0,:] = L_all_offset

    for lag in np.arange(ED1):
        X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[0,lag+1:]
        X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[0,lag+1:]
    
    for lag in np.arange(ED2):
        X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[0,:-lag-1]
        X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[0,:-lag-1]
        
      
    X3_go = np.zeros((ED2+1,np.size(Y,1)))
    X3_ng = np.zeros_like(X3_go)
    
    for st in stim_onset[(Xstim == 1)]:
        X3_go[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(Xstim ==0)]:
        X3_ng[0,st:st+stim_dur] = 1
        
    for lag in np.arange(ED2):
        X3_go[lag+1,lag+1:] = X3_go[0,:-lag-1]
        X3_ng[lag+1,lag+1:] = X3_ng[0,:-lag-1]
    
    
    X3_Hit = np.zeros((ED4+1,np.size(Y,1)))
    X3_FA = np.zeros_like(X3_Hit)
    X3_Miss = np.zeros_like(X3_Hit)
    X3_CR = np.zeros_like(X3_Hit)
       
    for st in stim_onset[(XHit==1)]:
        X3_Hit[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(XFA==1)]:
        X3_FA[0,st:st+stim_dur] = 1
            
            
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st:st+stim_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st:st+stim_dur] = 1 
        
        


    for lag in np.arange(ED4):
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]

    # gather X variables
    
    
    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    X3[4] = X3_Hit[0:10,:]
    X3[5] = X3_FA[0:10,:]
    X3[6] = X3_Miss[0:10,:]
    X3[7] = X3_CR[0:10,:]
    X3[8] = X3_Hit[10:,:]
    X3[9] = X3_FA[10:,:]
    X3[10] = X3_Miss[10:,:]
    X3[11] = X3_CR[10:,:]

    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = r_onset[1:150]
        Xstim = Xstim[1:150]
    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        Xstim = Xstim[200:D_ppc[n,4][0][0]+26]

    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
        Xstim = Xstim[D_ppc[n,4][0][0]+26:398]

        

    Y = Y[:,c1:c2]
    L_all = L_all[:,c1:c2]
    L_all_onset = L_all_onset[:,c1:c2]
    L_all_offset = L_all_offset[:,c1:c2]
    
    Y0 = np.mean(Y[:,:stim_onset[1]-50])
    
    for ind in np.arange(len(X3)):
        X3[ind] = X3[ind][:,c1:c2]         



    return X3,Y, L_all,L_all_onset, L_all_offset, stim_onset2, r_onset, Xstim,Y0

# %% glm_per_neuron function code
def glm_per_neuron(n,c_ind, fig_on,good_alpha):
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_Ca(D_ppc,n,window,c_ind)
    
    Y2 = Y # -Y0
    X4 = np.ones((1,np.size(Y)))
    alpha_list =  [1e-3,5*1e-3,1e-2,5*1e-2]
    # alpha_list =  [5*1e-2]
    l_ratio = 0.9
    alpha_score = np.zeros((len(alpha_list),1))
    aa = 0
    
    
    ### Iteration to find good alpha
    ### for run time purposes, run this code once with input good_alpha = 5*1e-2
    
    # good_alpha = 5*1e-2
    for alpha in alpha_list:
        reg = ElasticNet(alpha = alpha, l1_ratio = 0.9, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
        ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
    
        ### initial run, compare each TV ###
        Nvar= len(X)
        compare_score = {}
        int_alpha = 10
        for a in np.arange(Nvar+1):
            
            # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
    
            if a < Nvar:
                X4 = np.concatenate((X4,X[a]),axis = 0)
    
            cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
            compare_score[a] = cv_results['test_score']
        
        f = np.zeros((1,Nvar))
        p = np.zeros((1,Nvar))
        score_mean = np.zeros((1,Nvar))
        for it in np.arange(Nvar):
            f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
            score_mean[0,it] = np.median(compare_score[it])
    
        max_it = np.argmax(score_mean)
        init_score = compare_score[max_it]
        init_compare_score = compare_score
        
        if p[0,max_it] > 0.05:
                max_it = []
        else:  
                # === stepwise forward regression ===
                step = 0
                while step < Nvar:
                    max_ind = {}
                    compare_score2 = {}
                    f = np.zeros((1,Nvar))
                    p = np.zeros((1,Nvar))
                    score_mean = np.zeros((1,Nvar))
                    for it in np.arange(Nvar):
                        m_ind = np.unique(np.append(max_it,it))
                        # X4 = np.ones_like(Y)*int_alpha
                        X4 = np.zeros_like(Y)
                        for a in m_ind:
                            X4 = np.concatenate((X4,X[a]),axis = 0)
    
                        
                        cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                                    return_estimator = True, 
                                                    scoring = 'r2') 
                        compare_score2[it] = cv_results['test_score']
        
                        f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
                        score_mean[0,it] = np.mean(compare_score2[it])
                    max_ind = np.argmax(score_mean)
                    if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                        step = Nvar
                    else:
                        max_it = np.unique(np.append(max_it,max_ind))
                        init_score = compare_score2[max_ind]
                        step += 1
                        
                # === forward regression end ===
                # === running regression with max_it ===
                X3 = X
                if np.size(max_it) == 1:
                    max_it = [max_it,max_it]
                for tv_ind in [4,5,6,7]:
                    if (tv_ind+4 in max_it) and (tv_ind in max_it):
                        max_it = np.append(max_it, [tv_ind])
                            # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                        X3[tv_ind] = np.concatenate((X3[tv_ind],X3[tv_ind+4]),0);
                    elif (tv_ind+4 in max_it) and(tv_ind not in max_it):
                            max_it = np.append(max_it, [tv_ind])
                            X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                    elif (tv_ind+4 not in max_it) and(tv_ind in max_it):
                            # max_it = np.append(max_it, [tv_ind])
                            X3[tv_ind] = np.concatenate((X3[tv_ind],np.zeros_like(X3[tv_ind+4])),0);
                            
                            
                max_it = np.setdiff1d(max_it,[8,9,10,11])
                max_it = np.unique(max_it)
                
    
    
                # X4 = np.ones_like(Y)*int_alpha
                X4 = np.zeros_like(Y)
                # if np.size(max_it) == 1:
                #     X4 = np.concatenate((X4,X3[max_it]),axis = 0)
                # else:
                for a in max_it:
                        X4 = np.concatenate((X4,X3[a]),axis = 0)
                
                cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                            return_estimator = True, 
                                            scoring = 'r2') 
                score3 = cv_results['test_score']
                
                theta = [] 
                inter = []
                yhat = []
                for model in cv_results['estimator']:
                    theta = np.concatenate([theta,model.coef_]) 
                    # inter = np.concatenate([inter, model.intercept_])
                    yhat =np.concatenate([yhat, model.predict(X4.T)])
                    
                theta = np.reshape(theta,(k,-1)).T
                yhat = np.reshape(yhat,(k,-1)).T
                yhat = yhat + Y0
        alpha_score[aa,0] = np.mean(score3)
        aa += 1
    good_alpha = alpha_list[np.argmax(alpha_score)]
    ### iteration to find best alpha, end

    reg = ElasticNet(alpha = good_alpha, l1_ratio = l_ratio, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)

    ### initial run, compare each TV ###
    Nvar= len(X)
    compare_score = {}
    int_alpha = 10
    for a in np.arange(Nvar+1):
        
        # X4 = np.ones_like(Y)*int_alpha
        X4 = np.zeros_like(Y)

        if a < Nvar:
            X4 = np.concatenate((X4,X[a]),axis = 0)

        cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2') 
        compare_score[a] = cv_results['test_score']
    
    f = np.zeros((1,Nvar))
    p = np.zeros((1,Nvar))
    score_mean = np.zeros((1,Nvar))
    for it in np.arange(Nvar):
        f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
        score_mean[0,it] = np.median(compare_score[it])

    max_it = np.argmax(score_mean)
    init_score = compare_score[max_it]
    init_compare_score = compare_score
    
    if p[0,max_it] > 0.05:
            max_it = []
    else:  
            # === stepwise forward regression ===
            step = 0
            while step < Nvar:
                max_ind = {}
                compare_score2 = {}
                f = np.zeros((1,Nvar))
                p = np.zeros((1,Nvar))
                score_mean = np.zeros((1,Nvar))
                for it in np.arange(Nvar):
                    m_ind = np.unique(np.append(max_it,it))
                    # X4 = np.ones_like(Y)*int_alpha
                    X4 = np.zeros_like(Y)
                    for a in m_ind:
                        X4 = np.concatenate((X4,X[a]),axis = 0)

                    
                    cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'r2') 
                    compare_score2[it] = cv_results['test_score']
    
                    f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
                    score_mean[0,it] = np.mean(compare_score2[it])
                max_ind = np.argmax(score_mean)
                if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                    step = Nvar
                else:
                    max_it = np.unique(np.append(max_it,max_ind))
                    init_score = compare_score2[max_ind]
                    step += 1
                    
            # === forward regression end ===
            
        # === running regression with max_it ===
            X3 = X
            if np.size(max_it) == 1:
                max_it = [max_it,max_it]
            for tv_ind in [4,5,6,7]:
                if (tv_ind+4 in max_it) and (tv_ind in max_it):
                    max_it = np.append(max_it, [tv_ind])
                            # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                    X3[tv_ind] = np.concatenate((X3[tv_ind],X3[tv_ind+4]),0);
                elif (tv_ind+4 in max_it) and(tv_ind not in max_it):
                    max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                elif (tv_ind+4 not in max_it) and(tv_ind in max_it):
                            # max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = np.concatenate((X3[tv_ind],np.zeros_like(X3[tv_ind+4])),0);
                            
                            
            max_it = np.setdiff1d(max_it,[8,9,10,11])
            max_it = np.unique(max_it)
                
    
    
                # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
                # if np.size(max_it) == 1:
                #     X4 = np.concatenate((X4,X3[max_it]),axis = 0)
                # else:
            for a in max_it:
                X4 = np.concatenate((X4,X3[a]),axis = 0)
            
            cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
            score3 = cv_results['test_score']
            
            theta = [] 
            inter = []
            yhat = []
            for model in cv_results['estimator']:
                theta = np.concatenate([theta,model.coef_]) 
                yhat =np.concatenate([yhat, model.predict(X4.T)])
                
            theta = np.reshape(theta,(k,-1)).T
            yhat = np.reshape(yhat,(k,-1)).T
            yhat = yhat + Y0
    
    
    
    
    TT = {}
    lg = 1
    
    if np.size(max_it) ==1:
        a = np.empty( shape=(0, 0) )
        max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X3[t].T@theta[lg:lg+np.size(X3[t],0),:]  
            lg = lg+np.size(X3[t],0)
    except: 
        TT[max_it] = X3[max_it].T@theta[lg:lg+np.size(X3[max_it],0),:]  

    
    max_it = np.setdiff1d(max_it,[8,9,10,11])
    
    


    
    
    # === figure === 
    if fig_on ==1:
        prestim = 20
        t_period = 60
        
        y = np.zeros((t_period+prestim,np.size(stim_onset)))
        yh = np.zeros((t_period+prestim,np.size(stim_onset)))
        l = np.zeros((t_period+prestim,np.size(stim_onset))) 
        weight = {}
        for a in np.arange(Nvar):
           weight[a] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
        yhat_mean = np.mean(yhat,1).T - Y0    
        for st in np.arange(np.size(stim_onset)):
            y[:,st] = Y[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
            l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            # if np.size(max_it)>1:
            for t in max_it:
                weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)

        
        xaxis = np.arange(t_period+prestim)- prestim
        xaxis = xaxis*1e-1
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
        cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:red','tab:red','black','green']
        clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
        lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']
        
        ### plot y and y hat
        stim_ind1 = (Xstim ==1)
        stim_ind2 = (Xstim ==0)
    
        y1 = ndimage.gaussian_filter(np.mean(y[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(y[:,stim_ind2],1),0)
        s1 = np.std(y[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(y[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        y1 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind2],1),0)
        s1 = np.std(yh[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(yh[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "gray",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "gray",alpha = 0.5)
        
        
        
        ### plot model weights
        for a in np.arange(Nvar):
            y1 = ndimage.gaussian_filter(np.mean(weight[a],1),0)
            s1 = np.std(weight[a],1)/np.sqrt(np.size(weight[a],1))
            
            
            ax2.plot(xaxis,ndimage.gaussian_filter(y1,1),linewidth = 2.0,
                     color = cmap[a], label = clabels[a], linestyle = lstyles[a])
            ax2.fill_between(xaxis,(ndimage.gaussian_filter(y1,1) - s1),
                            (ndimage.gaussian_filter(y1,1)+ s1), color=cmap[a], alpha = 0.2)
        
        ### plot lick rate ###
        
        y1 = ndimage.gaussian_filter(np.mean(l[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(l[:,stim_ind2],1),0)
        s1 = np.std(l[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(l[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax3.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax3.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        
        ax2.set_title('unit_'+str(n+1))
        sc = np.mean(score3)
        ax4.set_title(f'{sc:.2f}')
        plt.show()
    
    
    return Xstim, L_on, inter, TT, Y, max_it, score3, init_compare_score, yhat,X4, theta, good_alpha

# %% Initialize
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 4000
prestim = 4000

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
list_a = np.load('list_alpha.npy',allow_pickle= True).item()

# %% Run GLM

Data = {}



for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in good_list: #np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            try:
                Xstim, L_on, inter, TT, Y, max_it, score3, init_score, yhat, X4, theta,g_alpha  = glm_per_neuron(n,c_ind,1,list_a[n])
                Data[n,c_ind-1] = {"X":Xstim,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                                    "intercept" : inter,'L' : L_on,"yhat" : yhat, "X4" : X4, "theta": theta,"alpha":g_alpha}
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Break, no fit") 
# np.save('RTnew_1211.npy', Data,allow_pickle= True)  

### load Data from saved file   
# Data = np.load('RTnew_1211.npy',allow_pickle= True).item()
# test = list(Data.keys())
# c_ind = c_list[0]
# good_list2 = np.zeros((len(test)))
# for n in np.arange(len(test)):
#     good_list2[n] =test[n][0]

# %% plot R score 


d_list3 = good_list2 <= 195

d_list = good_list2 > 195
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
Sstyles = ['tab:orange','none','tab:blue','none','tab:red','none','black','green','tab:purple','none']


def make_RS(d_list):
    good_list_sep = good_list2[d_list]
    ax_sz = len(cmap)-2
    I = np.zeros((np.size(good_list_sep),ax_sz+1))
       
        
    for n in np.arange(np.size(good_list_sep,0)):
        nn = int(good_list_sep[n])
        # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
        Model_score = Data[nn, c_ind-1]["score"]
        init_score =  Data[nn, c_ind-1]["init_score"]
        for a in np.arange(ax_sz):
            I[n,a] = np.mean(init_score[a])
        I[n,ax_sz] = np.mean(Model_score)
        
    
    fig, axes = plt.subplots(1,1, figsize = (10,8))
        # Rsstat = {}
    for a in np.arange(ax_sz):
        Rs = I[:,a]
        Rs = Rs[Rs>0.02]
        axes.scatter(np.ones_like(Rs)*(a+(c_ind+1)*-0.3),Rs,facecolors=Sstyles[a], edgecolors= cmap[a])
        axes.scatter([(a+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')    
            # Rs = Rs/(Rmax+0.03)
            # Rsstat[c_ind,f] = Rs
    
                # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    Rs = I[:,ax_sz]
    Rs = Rs[Rs>0.02]
    axes.scatter(np.ones_like(Rs)*(ax_sz+(c_ind+1)*-0.3),Rs,c = 'k',)
    axes.scatter([(ax_sz+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
    axes.set_ylim([0,0.75])
    axes.set_xlim([-1,len(cmap)])
    
    
    return I

I1 = make_RS(d_list3)
I2 = make_RS(d_list)
I1 = I1[:,8]
I2 = I2[:,8]
bins = np.arange(0,0.8, 0.01)
fig, axs= plt.subplots(1,1,figsize = (5,5))
axs.hist(I1[I1>0.02],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.hist(I2[I2>0.02],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.set_xlim([-.05,0.7])
# np.sum([I2>0.01])
# np.size(np.max(I1,1))
good_listRu = []       
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    Model_score = Data[nn, c_ind-1]["score"]
    if np.mean(Model_score) > 0.02:
        good_listRu = np.concatenate((good_listRu,[nn]))


# removing units with less than 2TTR
good_listRu = np.setdiff1d(good_listRu,[49,44,87,59,63])
good_listRu = np.setdiff1d(good_listRu,np.arange(117,130))
    

# %% helper functions to extract stim and reward onset 


def extract_onset_times(D_ppc,n):
    window = 100
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)
    
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]

    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = Rt-c1
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]



    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        # r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]

    return stim_onset2, r_onset

    
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    stim_onset,r_onset = extract_onset_times(D_ppc,nn)
    Data[nn,c_ind-1]["stim_onset"] = stim_onset
    Data[nn,c_ind-1]["r_onset"] = r_onset
    
    
# %% Normalized population average of task variable weights
d_list = good_listRu > 195
d_list3 = good_listRu <= 195

good_list_sep = good_listRu[:]

weight_thresh = 5*1e-2


# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green']
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']

# clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']
ax_sz = len(cmap)

w_length = [16,16,11,11,71,71,71,71] # window lengths for GLM 


Convdata = {}
Convdata2 = {}
pre = 25 # 10 40 
post = 55 # 50 20
xaxis = np.arange(post+pre)- pre
xaxis = xaxis*1e-1

for a in np.arange(ax_sz):
    Convdata[a] = np.zeros((np.size(good_list_sep),pre+post))
    Convdata2[a] = np.zeros(((np.size(good_list_sep),pre+post,w_length[a])))


good_list5 = [];
for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    Model_coef = Data[nn, c_ind-1]["coef"]
    theta = Data[nn,c_ind-1]["theta"]
    X4 = Data[nn,c_ind-1]["X4"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["r_onset"]
    # stim_onset= L_data[nn,1].T
    # stim_onset = stim_onset[0,1:-1]
    [T,p] = stats.ttest_1samp(np.abs(theta),0.05,axis = 1, alternative = 'greater') # set weight threshold here
    p = p<0.05
    Model_weight = np.multiply([np.mean(theta,1)*p],X4.T).T
    maxC2 = np.max([np.abs(np.mean(theta,1))*p])+0.2
    
    
    weight = {}
    weight2 = {}
    max_it = [key for key in Model_coef]
    # max_it = np.setdiff1d(max_it,[8,9,10,11])
    for a in max_it:
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
        weight2[a] = np.zeros((pre+post,np.size(stim_onset),w_length[a]) )  
                              
    for st in np.arange(np.size(stim_onset)-1):
        lag = 1
        for a in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15                
            
            weight[a][:,st] = np.mean(Model_coef[a][stim_onset[st]-pre: stim_onset[st]+post,:],1)
            weight2[a][:,st,:] = Model_weight[lag:lag+w_length[a],stim_onset[st]-pre: stim_onset[st]+post].T
                
            lag = lag+w_length[a]-1
        
    maxC = np.zeros((1,ax_sz))
    # [T,p] = stats.ttest_1samp(Model_score,0.01,alternative = 'greater')
    # if p < 0.05:
    #     good_list5 = np.concatenate((good_list5,[nn]))
    for a in max_it:    
            maxC[0,a] = np.max(np.abs(np.mean(weight[a],1)))+0.2
    for a in max_it:
            Convdata[a][n,:] = np.mean(weight[a],1) /np.max(maxC)
            # Convdata[a][n,:] = np.mean(weight[a],1) /(np.max(np.abs(np.mean(weight[a],1)))+0.2)
            nz_ind = np.abs(np.sum(weight2[a],(0,2)))>0
            if np.sum(nz_ind) > 0:
                if a == 6:
                    # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/(2*maxC2)
                    Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/2
                else:                       
                    # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/maxC2
                    Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)
        


### Figure : visualize average model weight    
# fig, axes = plt.subplots(1,1,figsize = (10,8))         
# axes.plot(xaxis,np.mean(weight[7],1))
# axes.plot(xaxis,np.mean(np.sum(weight2[a][:,nz_ind,:],1),1))
     
# fig, axes = plt.subplots(1,1,figsize = (10,8))       
# for a in [0]:
#     list0 = (np.mean(Convdata[a],1) != 0)
#     # error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
#     # y = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)   
#     W = ndimage.uniform_filter(np.sum(Convdata2[a][list0,:,:],2),[0,5], mode = "mirror")

#     # error = np.std(Convdata[a][list0,:],0)/np.sqrt(np.sum(list0))
#     # y = ndimage.gaussian_filter(np.mean(Convdata[a][list0,:],0),2)
#     y = np.abs(np.mean(W,0))
#     error = np.std(W,0)/np.sqrt(np.sqrt(np.sum(list0)))
#     axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
#     axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
#     axes.set_ylim([-0.01,0.25])
#     axes.set_ylim([-0.1,1])


# %% plotting weights by peak order
# Extended figure 7e
# Convdata2 = Model_weight
listOv = {}

f = 4
W5 = {}
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
b_count = {}
ax_sz = 8
w_length1 = [16,16,11,11,30,30,20,20]
w_length2 = [0,0,0,0,31,31,21,21]
for ind in [0,1]: # 0 is PPCIC, 1 is PPCAC
    b_count[ind] = np.zeros((2,ax_sz))

    for f in np.arange(ax_sz):
        W5[ind,f] = {}

for ind in [0,1]:
    for f in  np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        # list0 = (np.sum((Convdata[f],())
        # Lg = len(good_list2)
        Lg = len(good_listRu)
        Lic = np.where(good_listRu <194)
        Lic = Lic[0][-1]
        if ind == 0:
            list0[Lic:Lg] = False # PPCIC
        elif ind == 1:           
            list0[0:Lic] = False # PPCAC
        list0ind = good_listRu[list0]
        W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
        max_peak = np.argmax(np.abs(W),1)
        max_ind = max_peak.argsort()
        
        list1 = []
        list2 = []
        list3 = []
        
        SD = np.std(W[:,:])
        for m in np.arange(np.size(W,0)):
            n = max_ind[m]
            SD = np.std(W[n,:])
            if max_peak[n]> 0:    
                if W[n,max_peak[n]] >2*SD:
                    list1.append(m)
                    list3.append(m)
                elif W[n,max_peak[n]] <-2*SD:
                    list2.append(m)
                    list3.append(m)
                
        max_ind1 = max_ind[list1]  
        max_ind2 = max_ind[list2]     
        max_ind3 = max_ind[list3]
        max_peak3[ind,f] = max_peak[list3]
        
        listOv[ind,f] = list0ind[list3]
        
        W1 = W[max_ind1]
        W2 = W[max_ind2]    
        W4 = np.abs(W[max_ind3])
        s ='+' + str(np.size(W1,0)) +  '-' + str(np.size(W2,0))
        print(s)
        b_count[ind][0,f] = np.size(W1,0)
        b_count[ind][1,f] = np.size(W2,0)
        W3 = np.concatenate((W1,W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]

        W5[ind,f][0] = W1
        W5[ind,f][1] = W2
        if f in [7]: # use f index here to visualize model weights 
            clim = [-.5, .5]
            fig, axes = plt.subplots(1,1,figsize = (10,10))
            im1 = axes.imshow(W3[:,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W3
        elif ind == 1:           
            W5AC[f] = W3
        # W4IC = W4

# %% create list of all neurons that encode at least 1 variable
ind = 0
ax_sz = 8
test = [];
for ind in [0,1]:
    for f in np.arange(ax_sz):
        test = np.concatenate((test,listOv[ind,f]))

test_unique, counts = np.unique(test,return_counts= True)

# %% for each timebin, calculate the number of neurons encoding each TV
# Extended figure 7a,b, Figure 5f~k

cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green','tab:red','tab:orange','tab:purple','green',]
# cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']


Lic1 =np.argwhere(test_unique<194)[-1][0] +1 # 99 #134 # 78+1
Lg1 =len(test_unique)-Lic1
ind = 0# PPCIC or 1 PPCAC
p = 0# positive or 1 negative

fig, axes = plt.subplots(1,1,figsize = (10,5))
y_all = np.zeros((ax_sz,pre+post))
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
        
    Lg = len(good_list2)
    Lic = np.where(good_listRu <194)
    Lic = Lic[0][-1]
    if ind == 0:
        list0[Lic:Lg] = False # PPCIC
    elif ind == 1:           
        list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
    list0ind = good_listRu[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
    W = Convdata[f][list0,:]
    SD = np.std(W[:,:])
    # test = np.abs(W5[ind,f][p])>1*SD
    test = W5[ind,f][p]>2*SD
    if ind ==0:        
        y = np.sum(test,0)/Lic1
    elif ind == 1:
        y = np.sum(test,0)/Lg1
        
    y_all[f,:] = y
    y = ndimage.uniform_filter(y,2, mode = "mirror")
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = 'solid', linewidth = 3, label = clabels[f] )
        axes.set_ylim([0,.6])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = 'solid', linewidth = 3 )
        axes.set_ylim([-0.20,0])
        
    
plt.savefig("Fraction of neurons "+ ".svg")


# %% plot positive and negative weights separately.
# Extended figure 7c,d
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]

pp = 0
maxy = np.zeros((2,10))
for ind in [0,1]:
    fig, axes = plt.subplots(2,ax_sz,figsize = (50,10),sharex = "all")
    fig.subplots_adjust(hspace=0)
    for p in [0,1]:
        for f in np.arange(ax_sz):
            y1 = ndimage.gaussian_filter1d(np.sum(W5[ind,f][p],0),1)
            y1 = y1/(np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0))
            e1 = np.std(W5[ind,f][p],0)/np.sqrt((np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0)))
            axes[p,f].plot(xaxis,y1,c = cmap[f],linestyle = 'solid', linewidth = 3)
            axes[p,f].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
            # axes[p,f-3].set_xlim([-4,1])
            scat = np.zeros((2,np.size(W5IC[f],1)))
            pcat = np.zeros((2,np.size(W5IC[f],1)))
            maxy[p,f] = np.max(np.abs(y1)+np.abs(e1))
            maxy[p,f] = np.max([maxy[p,f],1])
            if np.size(W5[ind,f][p],0 > 4):
                for t in np.arange(80):
                    if p == 0:
                        s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],np.mean(e1),alternative = 'greater')
                    else:
                        s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],-np.mean(e1),alternative= 'less')
                    if p1 < 0.05:
                        scat[0,t] = True
                        pcat[0,t] = p1
                c1 = pcat[0,scat[0,:]>0]
                if p == 0:
                    axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*maxy[p,f] + 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])
                elif p ==1:
                    axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*-maxy[p,f] - 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])

    
    for f in np.arange(ax_sz):
            axes[0,f].set_ylim([0, np.nanmax(maxy[:,f]+0.2)])
            axes[1,f].set_ylim([-np.nanmax(maxy[:,f]+0.2),0])
            
# %% fraction of neurons, histogram
# Figure 5c 
Lic = 99
Lg = 202
b11 = b_count[0][0,:]/Lic
b12 = b_count[0][1,:]/Lic

b21 = b_count[1][0,:]/Lg
b22 = b_count[1][1,:]/Lg
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]

edgec = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]

axes.bar(np.arange(8)*3,b11+b12, color = 'white', edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2, hatch = '/')
axes.bar(np.arange(8)*3+1,b21+b22, color = cmap, edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2)

axes.set_ylim([0,0.9])


# %%  helper functions, load Rule1 and R2 data
Data1 = np.load('R1new_0718.npy',allow_pickle= True).item()

test = list(Data1.keys())
Y_R1 = {}
for t in np.arange(len(test)):
    n = test[t][0]
    Y_R1[n] = Data1[n,0]["Y"]
    
del Data1
    
Data2 = np.load('R2new_0718.npy',allow_pickle= True).item()
test = list(Data2.keys())
Y_R2 = {}
for t in np.arange(len(test)):
    n = test[t][0]
    Y_R2[n] = Data2[n,1]["Y"]
    
del Data2

YR = {}
YR[1] = Y_R1
YR[2] = Y_R2

def import_lick_mini(D_ppc,n,L2):
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    Lmax = np.max(np.floor(D_ppc[n,1]*(1e3/window)))
    if Lmax != L2:
        for l in np.floor(D_ppc[n,1]*(1e3/window)): 
            L_all[0,int(l[0])-1] = 1 
    
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    c1 = stim_onset[200]-100
    c2 = stim_onset[D_ppc[n,4][0][0]+26] 
    L = L_all[:,c1:c2]
    return L,Lmax


pre =  10
post =  70
o_ind = 0
xaxis = (np.arange(post+pre)- pre)*1e-1




def TVFR_ana(n,f):
    X = D_ppc[n,2][:,2:6] # task variables
    X = X[200:D_ppc[n,4][0][0]+15]
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    r_onset = Data[n, c_ind-1]["r_onset"]
    L = Data[n, c_ind-1]["L"]
    for l in np.arange(len(r_onset)):
        if r_onset[l] <0:
            r_onset[l] = stim_onset[l] + 15
    
    if o_ind == 1:
        stim_onset = r_onset

    X3 = {};
    X3[2] = (Xstim == 1) 
    X3[3] = (Xstim == 0)
    X3[4] = XHit
    X3[5] = XFA
    X3[6] = Xmiss
    X3[7] = XCR
    # dur = 20
    # dur2 = 50
    if f == 2:
        Xb = (Xstim == 1) 
    elif f == 3:
        Xb = (Xstim == 0)
    elif f == 4:
        Xb= XHit
        # Xb = XFA
    elif f== 6:
        Xb = Xmiss
    elif f == 5:
        Xb = XFA
        Xb = (Xstim == 0)
    elif f == 7:
        Xb = XCR
        Xb = (Xstim == 0)
    
    comp = np.zeros((len(X),pre+post))   
    comp_n = np.zeros((len(X),pre+post))
    Lc = np.zeros((len(X),pre+post))
    h = Data[n,c_ind-1]["Y"]
    for t in np.arange(len(X)):
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
                Lc[t,:] = L[0,stim_onset[t]-pre:stim_onset[t]+post]
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:], XFA[Xb], XCR[Xb],X3,Lc    

def import_data_mini(D_ppc,n,r,X):
    # N_trial = np.size(D_ppc[n,2],0)
    window = 100
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)
        
    
    ### Extract Ca trace ###

        
    if r == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        X2 = X[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]
    elif r == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        X2 = X[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
    try:
        Y = YR[r][n]    
    except:
        print(n)
        Yraw = {}
        Yraw = D_ppc[n,0]
        time_point = D_ppc[n,3]*1e3
        t = 0
        time_ind = []
        while t*window < np.max(time_point):
            time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
            t += 1
        
        
        Y = np.zeros((1,len(time_ind)-1))   
        for t in np.arange(len(time_ind)-1):
            Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
        Y = Y[:,c1:c2]
    
    return X2,Y, stim_onset2, r_onset
        

def TVFR_ana_exp(n,f,r):
    # pre= 10
    # post = 70
    X,Y,stim_onset, r_onset = import_data_mini(D_ppc,n,r,D_ppc[n,2][:,2:6])
    for l in np.arange(len(r_onset)):
        if r_onset[l] <0:
            r_onset[l] = stim_onset[l] + 15
    
    if o_ind == 1:
        stim_onset = r_onset

    # stim_onset = r_onset
    
    if r ==1:
        r_list = np.random.choice(np.arange(len(stim_onset)),50,replace = False)
        r_list = np.arange(20,120);
        r_list = np.arange(len(stim_onset))
    elif r ==2:
        
        r_list = np.arange(np.min([50,D_ppc[n,4][0][0]+15-200]))
        r_list = np.random.choice(np.arange(len(stim_onset)),50,replace = False)
        r_list = np.arange(20,70);
    X = X[r_list]
    stim_onset= stim_onset[r_list]                 
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    # dur = 20
    # dur2 = 50
    if r ==2:
        X3 = {};
        X3[2] = (Xstim == 1) 
        X3[3] = (Xstim == 0)
        X3[4] = XHit
        X3[5] = XFA
        X3[7] = XCR
        if f == 2:
            Xb = (Xstim == 1) 
        elif f == 3:
            Xb = (Xstim == 0)
        elif f == 4:
            Xb= XHit
        elif f == 5:
            Xb = XFA
            # Xb =XHit
        elif f == 6:
            Xb = Xmiss  
        elif f == 7:
            Xb = XCR
            Xb = (Xstim == 0)
    elif r == 1:
        X3 = {};
        X3[2] = (Xstim == 1) 
        X3[3] = (Xstim == 0)
        X3[4] = XHit
        X3[5] = XFA
        X3[6] = Xmiss
        X3[7] = XCR
        if f == 2:
            Xb = (Xstim == 0) 
        elif f == 3:
            Xb = (Xstim == 1)
        elif f == 4:
            Xb= XHit
            # Xb = XFA
        elif f == 5:
            Xb = XFA
            Xb = (Xstim == 0)*(X[:,1]==1) # FA trials
            # Xb = (Xstim == 1)*(X[:,1]==1) # Hit trials
        elif f == 6:
            Xb = Xmiss
        elif f == 7:
            Xb = XCR
            # Xb = (Xstim == 0)
        
        
    comp = np.zeros((len(X),pre+post))    
    comp_n = np.zeros((len(X),pre+post))
    h = Y
    for t in np.arange(len(X)):
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:], X3  



# %% Firing rate comparison during transition, from R1 to early to late
# figures 5g to 5p

f = 3
p = 1
comp = {}
comp_n = {}
comp_r1 = {}
comp_n_r1 = {}
comp_r2 = {}
comp_n_r2 = {}
comp_n = {}
comp_n[0] = {}
comp_n[1] = {}
comp_n[2] = {}

XCR = {};
XFA = {};
X3 = {}
X3[0] = {};
X3[1] = {};
X3[2] = {};
L = {};
for n in listOv[p,f]:
    nn = int(n)
    comp_n[0][nn], comp[nn], XFA[nn], XCR[nn],X3[0][nn],L[nn] = TVFR_ana(nn,f)
    comp_n[1][nn], comp_r1[nn], X3[1][nn] = TVFR_ana_exp(nn,f,1)
    comp_n[2][nn], comp_r2[nn], X3[2][nn] = TVFR_ana_exp(nn,f,2)
    
# %% this is the current final code
# divide up data for comparison
# Go, Nogo are divided into early vs late
# Hit, FA, CR are divided into R1, TR and R2
# Figure 6c~p

comp2= {};
comp2[0] = np.zeros((len(comp),pre+post))
comp2[1] = np.zeros((len(comp),pre+post))                    
comp2[2] = np.zeros((len(comp),pre+post))
comp2[3] = np.zeros((len(comp),pre+post))   
comp2[4] = np.zeros((len(comp),pre+post))   
comp_lick = np.zeros((len(comp),pre+post))

s_ind = 1
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    l = int(np.floor(len(comp[nn])/2))
    l2 = int(np.floor(len(comp[nn])/4))
    maxc = []
    minc = []
    for ind in [0,1,2]:
        for ind_f in [2,3,4,5,7]:
            if np.sum(X3[ind][nn][ind_f])>0:
                maxc = np.concatenate((maxc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),80)]))
                minc = np.concatenate((minc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),20)]))
    max_all = np.max(maxc)
    min_all = np.min(minc) 
    # min_all  = 0;

    if f in [0]:
            comp2[1][n,:] = (np.mean(comp[nn],0)-min_all)/(max_all-min_all+s_ind)
    else:
        if f in [3,5,7]:
            comp2[0][n,:] = (np.mean(comp[nn][XFA[nn],:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][XCR[nn],:],0)-min_all)/(max_all-min_all+s_ind)
        else:
            comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-min_all)/(max_all-min_all+s_ind)


    comp2[2][n,:] = (np.mean(comp_r1[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp2[3][n,:] = (np.mean(comp_r2[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp2[4][n,:] = (np.mean(comp[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp_lick[n,:] = np.mean(L[nn]*10,0)

# %% Outcome related variables analysis
if f == 7:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,5] == listOv[p,7][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
    
elif f == 5:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,7] == listOv[p,5][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
elif f == 4:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,4][[c]])  ==1:
        listind[0,c] = True 
    listind = (listind == 1)   

    listind2 = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,5] == listOv[p,4][[c]])  ==0:
            listind2[0,c] = True 
    listind2 = (listind2 == 1)  
    listind2 = listind2[0][listind[0]] 
    listind3= np.ones((10,len(listind2)))*listind2
elif f == 6: 
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,4][[c]])  ==1:
        listind[0,c] = True 
    listind = (listind == 1)  
        
        
W2 = {}
for ind in [0,1,2,3,4]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]
W = {};
if f ==5:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
# elif f == 4:
#     go_labels  = ['FA','Hit','R1','R2']
#     cmap = ['tab:orange','tab:red','black','grey']
elif f == 7:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
else:
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

if f in [4,5,6, 7]: #[4,5,7]:
    
    
    for ind in [0,1,2,3,4]:
        W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,1], mode = "mirror")
    max_peak = np.argmax(np.abs(W[1]),1)
    max_ind = max_peak.argsort()

    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,4))
    for ind in [0,1,2,3]:
        y = np.nanmean(W2[ind][listind[0],:],0)
        e = np.nanstd(W2[ind][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))

        axes.plot(xaxis,y,color= cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor= cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,pre+post))
        pcat[p_ind] = np.zeros((1,pre+post))
    s = {}
    pp = {}
    
    for t in np.arange(pre+post):
        s[0],pp[0] = stats.ttest_ind(W2[0][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')
        s[1],pp[1] = stats.ttest_ind(W2[1][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')      
        s[2],pp[2] = stats.ttest_ind(W2[0][listind[0],t], W2[3][listind[0],t],nan_policy = 'omit')
        for p_ind in [0,1,2]:
            pcat[p_ind][0,t] = pp[p_ind]
            if pp[p_ind] < 0.1:
                scat[p_ind][0,t] = True
                
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            # if p_ind == 2:
                # axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            if p_ind in [0,2]:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.3+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.legend()    
    axes.set_ylim([-0.,1])  

# %% rastermap for W2
from rastermap import Rastermap, utils

W3 = {}
for ind in [0,1,2,3,4]:
    W3[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,1],mode = "mirror")
ind =1
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=.75,
                  time_lag_window=7,
                  n_clusters = 10,
                  grid_upsample=1, keep_norm_X = False).fit(W3[ind])
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
# X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

for ind in [0,1,2,3]:
    fig, ax = plt.subplots(figsize = (5,5))
        # ax.imshow(zscore(W2[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
    ax.imshow(zscore(W3[ind][isort, :],axis = 1), cmap="gray_r", vmin=0.5, vmax=1.2,aspect = "auto")
    ax.set(xticks=np.arange(0, 80, 10), xticklabels=np.arange(-1, 7, 1));

# gather into variable W4 for PCA analysis
rng = np.random.default_rng()

W4 = {}
# ind 4 is RT all trials
for ind in [0,1,2,3,4]:
    W4[ind] = W3[ind]
 
W4[5] = np.zeros_like(W4[0])
for tn in np.arange(np.size(W3[0],0)):
    W4[5][tn,:] = W4[0][tn,:]
rng.shuffle(W4[5],axis = 0)

# %% stimulus related variable encoding analysis


peak = np.zeros((1,4))

if f == 3:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,0] == listOv[p,3][[c]])  == 0:
        listind[0,c] = True 
    listind = (listind == 1)  
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
    
    
    
if f == 2:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # xif np.sum(listOv[p,0] == listOv[p,2][[c]])  ==0:
        listind[0,c] = True 
    listind = (listind == 1)
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

W2 = {}
for ind in np.arange(5):
    W2[ind] = ndimage.uniform_filter(comp2[ind][listind[0],:],[0,1], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]


fig, axes = plt.subplots(1,1,figsize = (10,5))
            
for ind in [0,1,2,3]:
    y = np.nanmean(W2[ind],0)- np.nanmean(W2[2],0)
    e = np.nanstd(W2[ind]-W2[2],0)/np.sqrt(np.size(W2[ind],0))
    axes.plot(xaxis,y,color = cmap[ind],label = go_labels[ind])
    axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[ind],alpha = 0.3)
    peak[0,ind] = np.max(y)

scat = {}
pcat = {}
for p_ind in [0,1,2]:
    scat[p_ind] = np.zeros((1,pre+post))
    pcat[p_ind] = np.zeros((1,pre+post))
s = {}
pp = {}
for t in np.arange(pre+post):
    s[0],pp[0] = stats.ttest_ind(comp2[0][:,t], comp2[2][:,t])
    s[1],pp[1] = stats.ttest_ind(comp2[1][:,t], comp2[2][:,t])
    for p_ind in [0,1]:
        if pp[p_ind] < 0.1:
            scat[p_ind][0,t] = True
            pcat[p_ind][0,t] = pp[p_ind]
        c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
        if p_ind == 0:
            axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
        else:
            axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])

axes.legend()
axes.set_ylim([-0.2,0.4])    


W4 = {}
for ind in [0,1,2,3,4]:
    W4[ind] = ndimage.uniform_filter(W2[ind],[0,5], mode = "mirror")
    
    
# %% PCA on RT subspace. 
# Figure 6d,f,h,j,l,n,p

max_k = 20;

sm = 0
R = {}
t1 = 0
t2 = 35
pca = PCA(n_components=20)
R= W4[4][:,t1:t2].T
test = pca.fit_transform(ndimage.gaussian_filter(R,[1,0]))        
test = test.T


traj = {}
for ind in np.arange(len(W4)):
    traj[ind] = np.dot(W4[ind][:,t1:t2].T,pca.components_.T)
    traj[ind] = traj[ind] - np.mean(traj[ind][0:5,:],0)

# %%
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm


def draw_traj5(traj,v):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid','solid','dotted','solid','dotted','dashed','dashed']
    cmap_names = ['autumn','autumn','winter','winter']
    for ind in [0,1,2,3]: # np.arange(trmax):
            x = traj[ind][:,0]
            y = traj[ind][:,1]
            z = traj[ind][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,80))
                ax.auto_scale_xyz(x,y,z)
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,80))
                # ax.auto_scale_xyz(x,y,z)
            elif ind in [4]:
                colors = cm.winter(np.linspace(0,1,80))
            else :
                colors = cm.gray(np.linspace(0,1,80))

            

            lc = Line3DCollection(segments, color = colors,alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")


def traj_dist(array):
    array = array[:,[0,1,3]]
    distance  =np.zeros((1,t2-t1))
    for t in np.arange(t2-t1-1):
        distance[0,t] = np.linalg.norm(array[t+1,:]-array[t,:])
    return np.sum(distance)


draw_traj5(traj,0)



# %% trajectory distance with n iterated.
max_k = 20;

sm = 0
R = {}
t1 = 0
t2 = 20
total_n = np.size(W4[0],0)
n_cv = 50;
g = np.zeros((4,n_cv))
for cv in np.arange(n_cv):
    n_list = np.random.choice(np.arange(total_n),int(np.floor(total_n*0.75)),replace = False)       
    pca = PCA(n_components=20)
    R= W4[1][n_list,t1:t2].T
    test = pca.fit_transform(ndimage.gaussian_filter(R,[1,0]))        
    test = test.T
  
    traj = {}
    for ind in np.arange(len(W4)):
        traj[ind] = np.dot(W4[ind][n_list,t1:t2].T,pca.components_.T)
        traj[ind] = traj[ind] - np.mean(traj[ind][0:5,:],0)
    for g_ind in [0,1,2,3]:
        g[g_ind,cv] =traj_dist(traj[g_ind])
        
fig, axes= plt.subplots(1,1,figsize = (5,5))
axes.bar(np.arange(np.size(g,0)),np.mean(g,1))
axes.errorbar(np.arange(np.size(g,0)),np.mean(g,1),np.std(g,1), fmt="o", color="k", barsabove = True, capsize = 7, markersize = 0)
# axes.set_ylim([0,20])

stats.ranksums(g[0,:], g[1,:])
