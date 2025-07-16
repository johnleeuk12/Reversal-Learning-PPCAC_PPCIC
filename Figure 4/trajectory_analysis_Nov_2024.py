# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:34:17 2024

@author: Jong Hoon Lee
"""

# import packages 

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA

from os.path import join as pjoin
from numba import jit, cuda

from rastermap import Rastermap, utils
from scipy.stats import zscore

# %% File name and directory

fname = 'CaData_all_session_v3_corrected.mat'
fdir = 'D:\Python\Data'

# %% Helper functions for loading and selecting data
np.seterr(divide = 'ignore') 
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
    
    
    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y) > 0.5:
            good_list = np.concatenate((good_list,[n]))
    
    
    return good_list


def find_good_data_Ca2(t_period):
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


def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr):    
    
    L_all = np.zeros((1,int(np.floor(D_ppc[n,3][0,max(D_ppc[n,2][:,0])]*1e3))+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)

    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        stim_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,0]]*1e3))
        lick_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,3]]*1e3))
        lick_onset = lick_onset-stim_onset
        L[tr,:] = L_all[0,stim_onset-prestim-1:stim_onset+t_period-1]
        
        # reformatting lick rates
    L2 = []
    for w in range(int((t_period+prestim)/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l)) 
            
    L2 = np.reshape(L2,(int((t_period+prestim)/window),N_trial)).T


    X = D_ppc[n,2][:,2:6] # task variables
    Rt =  D_ppc[n,5] # reward time relative to stim onset, in seconds
    t_period = t_period+prestim
    
    # re-formatting Ca traces
    Yraw = {}
    Yraw = D_ppc[n,0]
    
    # Original Y calculation #####
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]
                
    # select analysis and model parameters with c_ind
    
    if c_ind ==0:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    elif c_ind == 1:        
        Y = Y[0:200,:]
        X = X[0:200,:]
        L2 = L2[0:200,:]
    elif c_ind ==2:
        Y = Y[D_ppc[n,4][0][0]:,:]
        X = X[D_ppc[n,4][0][0]:,:]
        L2 = L2[D_ppc[n,4][0][0]:,:]
    elif c_ind ==3:

        if ttr == -1:
            c1 = 200 
            c2 = D_ppc[n,4][0][0] +10
            # c1 = D_ppc[n,4][0][0]-20
            # c2 = D_ppc[n,4][0][0] +20
            
        elif ttr < 4:            
            c1 = ttr*50
            c2 = c1 +50
        else:
            c1 = D_ppc[n,4][0][0]+(ttr-4)*50
            # c1  = np.size(X,0)-200+(ttr-4)*50
            c2 = c1+ 50
            if ttr == 7:
                c2 = np.size(X,0)
            else:
                c2 = c1+50
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    Xpre2 = np.concatenate(([0,0],X[0:-2,2]*X[0:-2,1]),0)
    Xpre2 = Xpre2[:,None]
    # Add reward instead of action
    X2 = np.column_stack([X[:,0],X[:,3],
                          X[:,2]*X[:,1],Xpre]) 

    

    
    return X2,Y, L2, Rt

# %% Run main code
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 6000
prestim = 2000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 10 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



D_ppc = load_matfile_Ca()
good_list = find_good_data_Ca2(t_period)
    
# removing units with excessive drift or cut data
# good_list = np.delete(good_list,[365,366,367,368,369,370,371,372,373])
# %% get neural data
good_list = np.arange(len(D_ppc))
lenx = 160 # Length of data, 8000ms, with a 50 ms window.

D_all = np.zeros((len(good_list),lenx))
D = {}
trmax = 8
alpha = 5
for tr in np.arange(trmax):
    D[0,tr] = np.zeros((len(good_list),lenx))
    D[1,tr] = np.zeros((len(good_list),lenx))
    D[2,tr] = np.zeros((len(good_list),lenx))



for ind in [0,1,2]:
    D[ind,trmax] = np.zeros((len(good_list),lenx))
    D[ind,trmax+1] = np.zeros((len(good_list),lenx))

c_ind = 3
Y = {}
for tr in np.arange(trmax):
    print(tr)
    m = 0
    ttr = tr
    if tr == 4:
        ttr = -1
    elif tr >4:
        ttr = tr-1
        
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        for ind in [0,1]:
            D[ind,tr][m,:] = np.mean(Y[X[:,0] == ind,:],0)
            D[ind,tr][m,:] = D[ind,tr][m,:]/((np.max(np.mean(Y,0))) + alpha) # for original trajectories with Go+ NG
        m += 1
        
for c_ind in[1,2]:
    m = 0
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        for ind in [0,1]:
            D[ind,trmax+c_ind-1][m,:] = np.mean(Y[X[:,0] == ind,:],0)
            D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/(np.max(np.mean(Y,0)) + alpha)
        m += 1
        

c_ind =3
# %% sort units by peak order

d_list = good_list > 194 #PPCIC
d_list3 = good_list <= 194 # PPCAC


d_list2 = d_list
D_r = {}
sm = 5 # smoothing parameter rho
g_ind = 1 # group index, this is where data is sorted to 
trmax = 8 # max number of sections


for g in np.arange(trmax+2):
    D_r[g] = D[g_ind,g][d_list2,:]
    D_r[g] = ndimage.gaussian_filter(D_r[g],[sm,0])



max_ind = np.argmax(D_r[0][:,:],1)
max_peaks = np.argsort(max_ind)
clim = [-1,1.5]



fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))
for ind, ax in zip(np.arange(8), axs.ravel()):
    ax.imshow(zscore(D_r[ind][max_peaks, :],axis = 1),clim = clim, cmap="viridis", vmin=clim[0], vmax=clim[1], aspect="auto")


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(40, 15))
im1 =axs[0].imshow(zscore(D_r[8][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
im1 =axs[1].imshow(zscore(D_r[9][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
im1 =axs[2].imshow(zscore(D_r[4][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

# %% rastermap
# sorting using rastermap. Figure 4c and 4d

spks = D_r[4]

model = Rastermap(n_PCs=64,
                  locality=0.5,
                  time_lag_window=15,
                  n_clusters = 20,
                  grid_upsample=1, keep_norm_X = False).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)
clim = [-.5, 1.5]
    
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

axs[0].imshow(zscore((D_r[8][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
axs[1].imshow(zscore((D_r[4][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
axs[2].imshow(zscore((D_r[9][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
v = clim

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)


# %% PCA on individual groups for subspace overlap calculation

pca = {}
max_k = 20;
d_list = good_list > 195

d_list3 = good_list <= 195

trmax = 8

d_list2 = d_list3
fig, axs = plt.subplots(trmax+2,6,figsize = (20,30))

sm = 0
R = {}
test= {}
for g in  np.arange(trmax+2):
    pca[g] = PCA(n_components=20)
    R[g] = D[1,g][d_list2,:].T +D[0,g][d_list2,:].T
    test[g] = pca[g].fit_transform(ndimage.gaussian_filter(R[g],[2,0]))        
    test[g] = test[g].T
    for t in range(0,5):
        axs[g,t].plot(test[g][t,:], linewidth = 4)
    axs[g,5].plot(np.cumsum(pca[g].explained_variance_ratio_), linewidth = 4)
# %% subspace overlap calculation
# for explanation of method, see Bondanelli et al., 2021
from scipy import linalg

n_cv = 20   
trmax = 8


Overlap = np.zeros((trmax-1,trmax-1,n_cv)); # PPC_IC
Overlap_across = np.zeros((trmax,trmax,n_cv));


k1 = 10
k2 = 19

U = {}
for g in  np.arange(trmax+2):
    U[g], s, Vh = linalg.svd(R[g].T)

fig, axes = plt.subplots(1,1,figsize = (10,10))
for g1 in [0,1,2,4,5,6,7]: #np.arange(trmax):
   for g2 in [0,1,2,4,5,6,7]: # np.arange(trmax):
       S_value = np.zeros((1,k1))
       for d in np.arange(0,k1):
           S_value[0,d] = np.abs(np.dot(pca[g1].components_[d,:], pca[g2].components_[d,:].T))
           S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1].components_[d,:])*np.linalg.norm(pca[g2].components_[d,:]))

       if g1 < 4:
           if g2 < 4:
               Overlap[g1,g2,0] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1,g2-1,0] = np.max(S_value)
       elif g1 >= 4:
           if g2 < 4:
               Overlap[g1-1,g2,0] = np.max(S_value)
           elif g2 >= 4:
               Overlap[g1-1,g2-1,0] = np.max(S_value)
           
        
imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
imshowobj.set_clim(0.2, 0.8) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK









# %% plotting trajectories on common subspace

d_list = (good_list > 195)
d_list = np.logical_and(good_list > 195, good_list < 710,good_list > 720)

d_list3 = good_list <= 195

d_list2 = d_list3
sm = 5
# R =  np.empty( shape=(160,0) )
R =  np.empty( shape=(0,np.sum(d_list2)) )
label = np.empty( shape=(2,0) )
tr_max = 8
for tr in np.arange(tr_max):
    if tr != 3:
        for ind in [0,1]:
            R = np.concatenate((R,ndimage.gaussian_filter(D[ind,tr][d_list2,:].T,[sm,0])),0)
            # R = np.concatenate((R,zscore(D[ind,tr][d_list2,:].T,axis = 1)),0)
            lbl = np.concatenate((np.ones((1,np.sum(d_list2)))*tr,np.ones((1,np.sum(d_list2)))*ind),0)
            label = np.concatenate((label,lbl),1)



# RT = np.dot(R,raster_model.Usv)

pca = PCA(n_components=64)
test = pca.fit_transform(R)    

# %% draw trajectories


# from mpl_toolkits import mplot3d
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm


def draw_traj4(traj,v,trmax,sc,g):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid']
    cmap_names = ['autumn','winter','winter']
    for tr in [4,8,9]: # np.arange(trmax):
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,trmax))
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,trmax))

            lc = Line3DCollection(segments, color = colors[tr],alpha = 0.5,linestyle = styles[ind])#linestyle = styles[tr])
            if tr ==3:
                lc = Line3DCollection(segments, color = "purple", linestyle = styles[ind])
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
            # if tr == 0:
            #     ax.auto_scale_xyz(x,y,z)
    for tr in [trmax, trmax+1]:
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        
            colors = cm.copper(np.linspace(0,1,2))
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,2))
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,2))

            lc = Line3DCollection(segments, color = colors[tr-trmax],alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(4)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
                    
            # if tr == g and ind == 1:
            ax.auto_scale_xyz(x,y,z)
            # ax.set_xlim([-2,6])
            # ax.set_ylim([-1.5,1.5])
            # ax.set_zlim([-2,2])
            # PPCIC axes
            # ax.set_xlim([-1,5])
            # ax.set_ylim([-1.5,0.5])
            # ax.set_zlim([-2,1])
                
                

            
    if v ==1:
        for n in range(0, 100):
            if n >= 20 and n<50:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.elev = ax.elev+4.0 #pan down faster 
            if n >= 50 and n<80:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.elev = ax.elev+2.0 #pan down faster 
                ax.azim = ax.azim+4.0
            if n >= 80: #add axis labels at the end, when the plot isn't moving around
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
            # fig.suptitle(u'3-D Poincaré Plot, chaos vs random', fontsize=12, x=0.5, y=0.85)
            plt.gcf().set_size_inches(10, 10)
            # plt.savefig('charts/fig_size.png', )
            plt.savefig('Images/img' + str(n).zfill(3) + '.png',
                        dpi=200)
# %% PCA and trajectory space with grouped units



# from mpl_toolkits import mplot3d
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm


def draw_traj5(traj,v,g):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','dotted']
    cmap_names = ['autumn','winter','winter']
    for tr in [4,8,9]: # np.arange(trmax):
        for ind in [0,1]:
            x = traj[ind][tr][:,0]
            y = traj[ind][tr][:,1]
            z = traj[ind][tr][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,2))
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,2))

            
            if tr ==4:
                lc = Line3DCollection(segments, color = "purple", linestyle = styles[ind])
            else:
                lc = Line3DCollection(segments, color = colors[tr-8],alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")

# %%
traj = {}
R = {}

trmax = 7
g = 8

m = 0 
sm = 5
d_list2 = d_list
# d_list2  =d_list3

for ind in [0,1]:
    traj[ind] = {}
    for tr in [4,8,9]:
        traj[ind][tr] = np.dot(ndimage.gaussian_filter(D[ind,tr][d_list2,:].T,[sm,0]),pca[g].components_.T)

# for ind in [1,2,3]:
#         traj[ind] = ndimage.gaussian_filter(test.T,[sm,0])
#         traj[ind] = traj[ind]- np.mean(traj[ind][10:30,:],0)
#         m += 1


draw_traj5(traj,0,g)



# %% 
traj = {}
traj[0] = {}
traj[1] = {}
R = {}

trmax = 7

m = 0 
sm = 20
for tr in np.arange(trmax):
    for ind in [0,1]:
        traj[ind][tr] = ndimage.gaussian_filter(test[m*160:(m+1)*160,:],[sm,0])
        traj[ind][tr] = traj[ind][tr]- np.mean(traj[ind][tr][10:30,:],0)
        m += 1

for ind in [0,1]:
    traj[ind][trmax] = (traj[ind][0] + traj[ind][1] + traj[ind][2])/3
    traj[ind][trmax+1] = (traj[ind][4] + traj[ind][5] + traj[ind][6])/3



# trmax = 1
draw_traj4(traj,0,trmax,0,trmax)

