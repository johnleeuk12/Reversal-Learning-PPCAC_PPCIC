# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:09:16 2024

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

'''
For each neuron, get Y, neural data and X task variables.  
Stim onset is defined by stim onset time
Reward is defined by first lick during reward presentation
Lick onset, offset are defined by lick times
Hit vs FA are defined by trial conditions
for each Task variables, there's a set number of lag.
    
'''

def import_data_w_Ca(D_ppc,n,window,c_ind):    

    

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
    r_dur = 10 # 2 second reward duration 
    ED4 = 60

    
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
    
    for r in Rt[(XHit == 1)]:
        if r != 0:
            r = r-10
            X3_Hit[0,r:r+r_dur] = 1
    
    for r in Rt[(XFA == 1)]:
        if r != 0:
            r = r-10
            X3_FA[0,r:r+r_dur] = 1
            
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st+delay+stim_dur : st+delay+stim_dur+r_dur] = 1 
        
        


    for lag in np.arange(ED4):
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]
    
    
    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    X3[4] = X3_Hit
    X3[5] = X3_FA
    X3[6] = X3_Miss
    X3[7] = X3_CR

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

'''
We model each neuron's FR Y with X, using stepwise regression for each task variable (but not for each lag)

'''
def glm_per_neuron(n,c_ind, fig_on):
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_Ca(D_ppc,n,window,c_ind)
    
    Y2 = Y # -Y0
    #Using a linear regression model with Ridge regression regulator set with alpha = 1e-3
    reg = ElasticNet(alpha = 1e-3, l1_ratio = 0.9, fit_intercept=True) 
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
            
            # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
            if np.size(max_it) == 1:
                X4 = np.concatenate((X4,X[max_it]),axis = 0)
            else:
                for a in max_it:
                    X4 = np.concatenate((X4,X[a]),axis = 0)
            
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
    
    TT = {}
    lg = 1
    
    if np.size(max_it) ==1:
        a = np.empty( shape=(0, 0) )
        max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X[t].T@theta[lg:lg+np.size(X[t],0),:]  
            lg = lg+np.size(X[t],0)
    except: 
        TT[max_it] = X[max_it].T@theta[lg:lg+np.size(X[max_it],0),:]  
    
    
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
            # else:
            #     weight[max_it][:,st] = np.mean(TT[max_it][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            
    
        
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
    
    
    return Xstim, L_on, inter, TT, Y, max_it, score3, init_compare_score, yhat,X4, theta



# %% Initialize (This is where the analysis starts)
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
c_list = [2]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
# %% Run GLM

Data = {}



for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in good_list: 
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            try:
                Xstim, L_on, inter, TT, Y, max_it, score3, init_score, yhat, X4, theta  = glm_per_neuron(n,c_ind,1)
                Data[n,c_ind-1] = {"X":Xstim,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                                    "intercept" : inter,'L' : L_on,"yhat" : yhat, "X4" : X4, "theta": theta}
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Break, no fit") 
# np.save('R2new_0718.npy', Data,allow_pickle= True)  

# %% plot R score 


d_list3 = good_list2 <= 195 # list for PPCIC

d_list = good_list2 > 195 # list for PPCAC
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
        I[n,ax_sz] = np.mean(Model_score)*1.5
        
    
    fig, axes = plt.subplots(1,1, figsize = (10,8))
        # Rsstat = {}
    for a in np.arange(ax_sz):
        Rs = I[:,a]
        Rs = Rs[Rs>0.01]
        axes.scatter(np.ones_like(Rs)*(a+(c_ind+1)*-0.3),Rs,facecolors=Sstyles[a], edgecolors= cmap[a])
        axes.scatter([(a+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')    

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
axs.hist(I1[I1>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.hist(I2[I2>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)


# %% helper functions for extracting onset times, both stim and reward onset


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

# add onset times to Data dict
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    stim_onset,r_onset = extract_onset_times(D_ppc,nn)
    Data[nn,c_ind-1]["stim_onset"] = stim_onset
    Data[nn,c_ind-1]["r_onset"] = r_onset 
    
# %% Normalized population average of task variable weights
d_list = good_list2 > 195
d_list3 = good_list2 <= 195

good_list_sep = good_list2[:]

weight_thresh = 5*1e-2


# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']
ax_sz = len(cmap)

w_length = [16,16,11,11,61,61,61,61] # window lengths for GLM 


Convdata = {}
Convdata2 = {}
pre = 10 # 10 40 
post = 70 # 50 20
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
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    
    [T,p] = stats.ttest_1samp(np.abs(theta),0.05,axis = 1, alternative = 'greater') # set weight threshold here
    p = p<0.05
    Model_weight = np.multiply([np.mean(theta,1)*p],X4.T).T
    maxC2 = np.max([np.abs(np.mean(theta,1))*p])+0.2
    
    
    weight = {}
    weight2 = {}
    max_it = [key for key in Model_coef]
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
    for a in max_it:    
            maxC[0,a] = np.max(np.abs(np.mean(weight[a],1)))+0.5
    for a in max_it:
            Convdata[a][n,:] = np.mean(weight[a],1) /np.max(maxC)
            nz_ind = np.abs(np.sum(weight2[a],(0,2)))>0
            if np.sum(nz_ind) > 0:
                Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/maxC2
        
fig, axes = plt.subplots(1,1,figsize = (10,8))       
for a in np.arange(ax_sz):
    error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
    y = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)
    # y = np.abs(y)
    axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
    axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
    axes.set_ylim([-0.01,0.25])
    
    
# %% plotting weights by peak order

listOv = {}

f = 5
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
    for f in np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        # list0 = (np.sum((Convdata[f],())
        Lg = len(good_list2)
        Lic = np.where(good_list2 <194)
        Lic = Lic[0][-1]
        if ind == 0:
            list0[Lic:Lg] = False # PPCIC
        elif ind == 1:           
            list0[0:Lic] = False # PPCAC
        

        list0ind = good_list2[list0]
        W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,:],2),[0,0], mode = "mirror")
        W = W/int(np.floor(w_length1[f]/10)+1)
        max_peak = np.argmax(np.abs(W),1)
        max_ind = max_peak.argsort()
        
        list1 = []
        list2 = []
        list3 = []
        
        SD = np.std(W[:,:])
        for m in np.arange(np.size(W,0)):
            n = max_ind[m]
            SD = np.std(W[n,:])
            # if SD< 0.05:
            #     SD = 0.05
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
        W3 = np.concatenate((W1,-W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]
        W3[:,0:8] = 0
        W5[ind,f][0] = W3
        W5[ind,f][1] = W3
        if f in [7]:
            clim = [-0.7, 0.7]
            fig, axes = plt.subplots(1,1,figsize = (10,10))
            im1 = axes.imshow(W3[:,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W3
        elif ind == 1:           
            W5AC[f] = W3

# %% create list of all neurons that encode at least 1 variable

ind = 0
ax_sz = 8
test = [];
for ind in [0,1]:
    for f in np.arange(ax_sz):
        test = np.concatenate((test,listOv[ind,f]))

test_unique, counts = np.unique(test,return_counts= True)

# %% plot each weights 
# fig 5e
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']


pp = 3
maxy = np.zeros((2,10))
fig, axes = plt.subplots(1,1,figsize = (10,5),sharex = "all")
fig.subplots_adjust(hspace=0)
for f in [pp]: #np.arange(ax_sz):
    W5IC[f][-4:,:] = 0
    for ind in [0,1]:
        if ind == 0:
            y1 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),1)
            e1 = np.std(W5IC[f],0)/np.sqrt(np.size(W5IC[f],0))
        elif ind ==1:
            y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),1)
            e1 = np.std(W5AC[f],0)/np.sqrt(np.size(W5AC[f],0))
        axes.plot(xaxis,y1,c = cmap[f],linestyle = lstyles[ind+1], linewidth = 3)
        axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
    # ks test
    scat = np.zeros((2,np.size(W5IC[f],1)))
    pcat = np.zeros((2,np.size(W5IC[f],1)))
    for t in np.arange(np.size(W5IC[f],1)):
        s1,p1 = stats.ks_2samp(W5IC[f][:,t], W5AC[f][:,t],'less')
        s2,p2 = stats.ks_2samp(W5AC[f][:,t], W5IC[f][:,t],'less')
        if p1 < 0.05:
            scat[0,t] = True
            pcat[0,t] = p1
        if p2 < 0.05:
            scat[1,t] = True
            pcat[1,t] = p2
    c1 = pcat[0,scat[0,:]>0]
    c2 = pcat[1,scat[1,:]>0]
    axes.scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*0.85,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.scatter(xaxis[scat[1,:]>0],np.ones_like(xaxis[scat[1,:]>0])*0.85,marker='s',c = np.log10(c2),cmap = 'Greys_r',clim = [-3,0])

        
    axes.set_ylim([-0.05,0.9])

# %% for each timebin, calculate the number of neurons encoding each TV
# fig5
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']


Lic1 =np.argwhere(test_unique<194)[-1][0] +1 
Lg1 =len(test_unique)-Lic1
ind =1 # PPCIC or 1 PPCAC
p = 0 # positive or 1 negative

fig, axes = plt.subplots(1,1,figsize = (10,5))
y_all = np.zeros((ax_sz,80))
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
        
    Lg = len(good_list2)
    Lic = np.where(good_list2 <194)
    Lic = Lic[0][-1]
    if ind == 0:
        list0[Lic:Lg] = False # PPCIC
    elif ind == 1:           
        list0[0:Lic] = False # PPCAC
        
    list0ind = good_list2[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
    W = Convdata[f][list0,:]
    SD = np.std(W[:,:])
    test = W5[ind,f][p]>2*SD
    if ind ==0:        
        y = np.sum(test,0)/Lic1
    elif ind == 1:
        y = np.sum(test,0)/Lg1
        
    y_all[f,:] = y
    y = ndimage.uniform_filter(y,2, mode = "mirror")
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = 'solid', linewidth = 3, label = clabels[f] )
        axes.set_ylim([0,.3])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = 'solid', linewidth = 3 )
        axes.set_ylim([-0.20,0])
        

# plt.savefig("Fraction of neurons "+ ".svg")
# %% fraction of neurons, histogram
# fig 5c 
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

# %% helper functions, load Rule1 and R2 data
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


pre= 30
post = 50
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
    
    for l in np.arange(len(r_onset)):
        if r_onset[l] <0:
            r_onset[l] = stim_onset[l] + 15
            
    stim_onset = r_onset
    # dur = 20
    # dur2 = 50
    if f == 2:
        Xb = (Xstim == 1) 
    elif f == 3:
        Xb = (Xstim == 0)
    elif f == 4:
        Xb= XHit
        Xb = (Xstim == 1) 
    elif f == 5:
        Xb = XFA
        Xb = (Xstim == 0)
    elif f == 7:
        Xb = XCR
        Xb = (Xstim == 0)
    
    comp = np.zeros((len(X),80))   
    comp_n = np.zeros((len(X),80))
    h = Data[n,c_ind-1]["Y"]
    for t in np.arange(len(X)):
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:], XFA[Xb], XCR[Xb]    


def import_data_mini(D_ppc,n,r,X):
    N_trial = np.size(D_ppc[n,2],0)
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
    stim_onset = r_onset
    
    if r ==1:
        r_list = np.random.choice(np.arange(len(stim_onset)),50,replace = False)
    elif r ==2:
        
        r_list = np.arange(np.min([50,D_ppc[n,4][0][0]+15-200]))
        
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
        if f == 2:
            Xb = (Xstim == 1) 
        elif f == 3:
            Xb = (Xstim == 0)
        elif f == 4:
            Xb= XHit
        elif f == 5:
            Xb = XFA
        elif f == 7:
            Xb = XCR
            Xb = (Xstim == 0)
    elif r == 1:
        if f == 2:
            Xb = (Xstim == 0) 
        elif f == 3:
            Xb = (Xstim == 1)
        elif f == 4:
            Xb= XHit
        elif f == 5:
            Xb = XFA
            Xb = (Xstim == 0)
        elif f == 7:
            Xb = XCR
            Xb = (Xstim == 0)
        
        
    comp = np.zeros((len(X),80))    
    comp_n = np.zeros((len(X),80))
    h = Y
    for t in np.arange(len(X)):
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:]    


def TVFR_ana_r2(n,f):
    # pre= 10
    # post = 70
    X = D_ppc[n,2][:,2:6] # task variables
    X = X[200:D_ppc[n,4][0][0]+26]
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    # stim_onset= stim_onset[D_ppc[n,4][0][0]-200:]
    # dur = 20
    # dur2 = 50
    
    Xb1 = {}
    if f == 2:
        Xb = (Xstim == 1) 
    elif f == 3:
        Xb = (Xstim == 0)
    elif f == 4:
        Xb= XHit
    elif f == 5:
        Xb = XFA
    elif f == 7:
        # Xb = XCR
        Xb = (Xstim == 0)

    
    
    comp = np.zeros((len(X),80))    

    h = Data[n,c_ind-1]["Y"]
    for t in np.arange(len(X)):
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    
    return comp[Xb,:]    
# %% Firing rate comparison during transition, from R1 to early to late
# figures 5g to 5p

# select area and task-variable
f = 4
p = 1
comp = {}
comp_n = {}
comp_r1 = {}
comp_n_r1 = {}
comp_r2 = {}

XCR = {};
XFA = {};
for n in listOv[p,f]:
    nn = int(n)
    comp_n[nn], comp[nn], XFA[nn], XCR[nn] = TVFR_ana(nn,f)
    comp_n_r1[nn], comp_r1[nn] = TVFR_ana_exp(nn,f,1)
    # comp_r2[nn] = TVFR_ana_exp(nn,f,2)
    # comp_r2[nn] = TVFR_ana_r2(nn,f)


# %%
comp2= {};
comp2[0] = np.zeros((len(comp),80))
comp2[1] = np.zeros((len(comp),80))                    
comp2[2] = np.zeros((len(comp),80))
comp2[3] = np.zeros((len(comp),80))   

s_ind = 1
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    l = int(np.floor(len(comp[nn])/2))
    l2 = int(np.floor(len(comp[nn])/4))
    maxc = np.percentile(np.mean(comp_n[nn],0),90)
    minc = np.percentile(np.mean(comp_n[nn],0),10)
    
    if f in [3,5,7]: #[3,5,7]:

            comp2[0][n,:] = (np.mean(comp[nn][XFA[nn],:],0)-minc)/(maxc-minc+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][XCR[nn],:],0)-minc)/(maxc-minc+s_ind)
    else:
            comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-minc)/(maxc-minc+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-minc)/(maxc-minc+s_ind)


    # 
    maxc = np.percentile(np.mean(comp_n_r1[nn],0),90)
    minc = np.percentile(np.mean(comp_n_r1[nn],0),20)

    comp2[2][n,:] = (np.mean(comp_r1[nn],0)-minc)/(maxc-minc+s_ind)


if f == 0:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,2] == listOv[p,3][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1) 



elif f == 4:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,2] == listOv[p,4][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)        
        

W2 = {}
for ind in [0,1,2]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]
                    



# rastermap
from rastermap import Rastermap
from scipy.stats import zscore

W3 = {}
for ind in [0,1,2]:
    W3[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,3],mode = "mirror")
                              
    
ind = 1
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=0.75,
                  time_lag_window=5,
                  n_clusters = 20,
                  grid_upsample=5, keep_norm_X = False).fit(W3[ind])
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
# X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

for ind in [0,1]:
    fig, ax = plt.subplots(figsize = (5,5))
        # ax.imshow(zscore(W2[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
    ax.imshow(zscore(W3[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")
        # ax.imshow(W3[ind][isort, :], cmap="gray_r", aspect="auto")

# rastermpa end

W = {};
if f in [3,5,7]:
    go_labels  = ['FA','CR','R1']
    cmap = ['tab:orange','tab:green','black','grey']
else:
    go_labels  = ['Early','Late','R1']
    cmap = ['tab:blue','tab:orange','black','grey']

if f in [3, 4, 7]: #[4,5,7]:
    
    
    for ind in [0,1,2]:
        W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,2], mode = "mirror")
        # W[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W[0]),1)
    max_ind = max_peak.argsort()    
    for ind in [0,1,2]:
        fig, axes = plt.subplots(1,1,figsize = (5,5))        
        # im1 = axes.imshow(W[ind][max_ind,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")\
        im1 = axes.imshow(zscore(W[ind][max_ind, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,3))
    for ind in [0,1,2]:
        # y = np.mean(comp2[ind][listind[0],:],0)
        # e = np.std(comp2[ind][listind[0],:],0)/np.sqrt(np.size(comp2[ind],0))
        y = np.nanmean(W2[ind][listind[0],:],0)
        e = np.nanstd(W2[ind][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))
        axes.plot(xaxis,y,color= cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor= cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,80))
        pcat[p_ind] = np.zeros((1,80))
    s = {}
    pp = {}
    
    for t in np.arange(80):
            # s1,p1 = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
        s[0],pp[0] = stats.ttest_ind(W2[0][listind[0],t], W2[1][listind[0],t])
        s[1],pp[1] = stats.ttest_ind(W2[0][listind[0],t], W2[2][listind[0],t])
        s[2],pp[2] = stats.ttest_ind(W2[1][listind[0],t], W2[2][listind[0],t])
        for p_ind in [0,1,2]:
            if pp[p_ind] < 0.05:
                scat[p_ind][0,t] = True
                pcat[p_ind][0,t] = pp[p_ind]
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            if p_ind == 0:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            else:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.legend()    
    # axes.set_ylim([-0.1,1.0])    
else:
    
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,3))
    for ind in [0,1,2]:
        y = np.nanmean(comp2[ind],0)
        e = np.nanstd(comp2[ind],0)/np.sqrt(np.size(comp2[ind],0))
        axes.plot(xaxis,y,color = cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)

    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,80))
        pcat[p_ind] = np.zeros((1,80))
    s = {}
    pp = {}
    for t in np.arange(80):
        s[0],pp[0] = stats.ttest_ind(comp2[0][:,t], comp2[1][:,t])
        s[1],pp[1] = stats.ttest_ind(comp2[0][:,t], comp2[2][:,t])
        s[2],pp[2] = stats.ttest_ind(comp2[1][:,t], comp2[2][:,t])
        # s1,p1 = stats.ks_2samp(comp2[0][:,t], comp2[1][:,t])
        for p_ind in [0,1,2]:
            if pp[p_ind] < 0.05:
                scat[p_ind][0,t] = True
                pcat[p_ind][0,t] = pp[p_ind]
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            if p_ind == 0:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            else:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])

    axes.legend()
    axes.set_ylim([-0.1,1.0])    

    for ind in [0,1,2]:
        W[ind] = ndimage.uniform_filter(W2[ind],[0,2], mode = "mirror")
        # W[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W[1]),1)
    max_ind = max_peak.argsort()    
    for ind in [0,1,2]:
        fig, axes = plt.subplots(1,1,figsize = (5,5))        
        im1 = axes.imshow(W[ind][max_ind,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        
        
