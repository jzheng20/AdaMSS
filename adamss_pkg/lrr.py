import torch
import scipy.io
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

 


def slicePCA(tensor,r,device,dtype=torch.float32):  
    tensor=tensor.to(device)
    B, C, H, W = tensor.shape 
    #print(tensor.shape) 
    UU=torch.zeros(B, C, H,r, dtype=dtype,device=device) 
    VVT=torch.zeros(B, C, r,W, dtype=dtype,device=device)  
    for i in range(B):
        for j in range(C): 
            #U, S, VT = torch.linalg.svd(tensor[i, j, :, :], full_matrices=False)  # More stable than torch.svd
            U,_,V=torch.svd_lowrank(tensor[i, j, :, :], q=r, niter=2, M=None)
            UU[i,j,:,:]=U[:,0:r] 
            #VVT[i,j,:,:]=VT[0:r,:]  
            VVT[i,j,:,:]=V[:,0:r].T
    return VVT, UU



def tensor_T(X,device):
    XT=torch.zeros(X.shape[0],X.shape[1],X.shape[3],X.shape[2],device=device) 
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XT[i,j,:,:]=X[i,j,:,:].T
    return XT

def clustering_Z(VT,K,iternum):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
     
    KK=[]  
    indxx=[]
    for i in range(1): 
        kmeans = KMeans(n_clusters=K, init='random', n_init=1, max_iter=iternum, random_state=123456789)
        idx = kmeans.fit_predict(VT.cpu().numpy())
        indxx.append(idx)
        KK.append(K)
    return indxx, KK
 


def seg_locations(WW_shape0,index):
    locations=[]
    for ii in range(WW_shape0): 
        # each collumn in W is a sample
        K = index[ii].max().item()+1
        location=[]
        for i in range(K):
            location.append(np.where(index[ii] == i)[0]) 
        locations.append(location)
    return locations



def get_trainable_subspaces_all(num_layers,KK):
    top_indicess=[]
    for ii in range(num_layers):
        top_indices=[]
        for i in range(KK[ii]):
            top_indices.append(i)
        top_indicess.append(top_indices)
    return top_indicess
