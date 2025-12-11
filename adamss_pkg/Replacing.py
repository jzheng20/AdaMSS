import torch.nn as nn
#from .TensorTools.TensorSVD import *
from .utilized import * 
from .lrr1 import *


 

class adamss_layer(nn.Module):
    def __init__(self, resW,A,B,newA,newB, KK,TrainSubsp_indx,seg_result,bias, TrainB=True):
        super(adamss_layer, self).__init__() 
        self.adamss_resW =torch.Tensor(resW).float()
        self.KK=KK
        self.TrainSubsp_indx=TrainSubsp_indx
        self.seg_result=seg_result
        self.newB=newB
        for i in range(self.KK):
            indx=self.TrainSubsp_indx[i] 
            setattr(self,f'adamss_A_{i}',A[i])
            setattr(self,f'adamss_B_{i}',B[i]) 
            setattr(self,f'adamss_newA_{i}',torch.Tensor(newA[seg_result[indx],:]).float()) 
          
        if TrainB==None:
            self.bias = torch.Tensor(bias).float() 
        else:  
            self.bias=nn.Parameter(torch.Tensor(bias).float())

    def forward(self,x):
        seg_result=self.seg_result 
        x1=nn.functional.linear(x, self.adamss_resW) 
        x4=torch.zeros_like(x)
        x20=nn.functional.linear(x,self.newB) 
        for i in range(self.KK):
            indx=self.TrainSubsp_indx[i] 
            A=getattr(self,f'adamss_A_{i}')
            B=getattr(self,f'adamss_B_{i}')
            newA=getattr(self,f'adamss_newA_{i}') 
            x30=nn.functional.linear(x20,newA) 
            x40=nn.functional.linear(x30,A.T) 
            if self.bias is not None:  
                x4[:,:,seg_result[indx]]=nn.functional.linear(x40,B.T,self.bias[seg_result[indx]]) 
            else: 
                x4[:,:,seg_result[indx]]=nn.functional.linear(x40,B.T) 
        return x1+x4 

class adamss_layer_bias(nn.Module):
    def __init__(self, resW,A,B,newA,newB,KK,TrainSubsp_indx,seg_result,dtype):
        super(adamss_layer_bias, self).__init__() 
        self.dtype=dtype #torch.float32
        self.adamss_resW =torch.Tensor(resW).to(dtype)
        self.KK=KK
        self.TrainSubsp_indx=TrainSubsp_indx
        self.seg_result=seg_result 
        self.newB=newB.to(dtype) 
        self.device=newA.device
        for i in range(self.KK):
            indx=TrainSubsp_indx[i] 
            Q, R =torch.linalg.qr((newA[seg_result[TrainSubsp_indx[i]],:]).T@A[i], mode='reduced')
            setattr(self,f'adamss_A_{i}',nn.Parameter(Q.T.to(dtype))) 
            setattr(self, f'adamss_B_{i}', nn.Parameter(torch.zeros_like(B[i].T).to(dtype)))
               
        self.newindex=np.concatenate([self.seg_result[self.TrainSubsp_indx[i]] for i in range(self.KK)], axis=0)
       


    
    def forward(self,x): 
        x=x.to(self.dtype)
        stacked_A = torch.cat([getattr(self, f'adamss_A_{i}') for i in range(self.KK)], dim=0)
        stacked_B = torch.block_diag(*[getattr(self, f'adamss_B_{i}') for i in range(self.KK)])
        ones = torch.ones(x.shape[0], x.shape[1], 1, device=x.device,dtype=self.dtype)
        newx = torch.cat((x, ones), dim=-1)
        x1=nn.functional.linear(newx, self.adamss_resW) 
        
        x2=nn.functional.linear(newx,self.newB) 
 
   
        x5=nn.functional.linear(x2,stacked_A)
        x6=nn.functional.linear(x5,stacked_B)
        #x7=x6@self.maskT.to_dense() 
        x7=torch.zeros_like(x6) 
        x7[:,:,self.newindex]=x6
         
        return  x1+x7

 
 
 
class replacing_with_AdaMSS(nn.Module):
    def __init__(self,args, index,  multi_heads,multi_heads_fa, FoDs_size,qkv_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(replacing_with_AdaMSS, self).__init__()
        print("=========Start Initialization=========")
 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype=None
        print(device)
        self.with_bias=True
        self.index=index
        self.device=device 
        self.qkv_name=qkv_name  
        self.FoDs_size=FoDs_size
        self.transform_type=args.transform_type
        if self.with_bias is True:
            weightTensor=self.get_tensor_bias(multi_heads, FoDs_size)
        else:
            weightTensor=self.get_tensor(multi_heads, FoDs_size)
        weightTensor=torch.tensor(weightTensor).to(self.device)
        print("================Subspace Segmentation==========")
        indx,ZTensor,newWeights,self.KK,newA,newB=seg_tensor(weightTensor,args.adamss_R,args.adamss_K,args.adamss_EK,0.01)  
        print("==============Matrix Decomposition==============")
        self.newA,self.newB=torch.tensor(newA).to(self.device),torch.tensor(newB).to(self.device)
        newWeights=torch.tensor(newWeights).to(self.device)
        self.newWeights=newWeights
 
        ZTensor=torch.tensor(ZTensor).to(self.device)
        seg_results=seg_locations(newWeights,indx)
        #if args.trainable_subspaces == 'all':
        TrainSubsp_indx=get_trainable_subspaces_all(FoDs_size['num_layers'],self.KK)
        #else:
            #TrainSubsp_indx=get_trainable_subspaces(trainable_rows,seg_results,self.KK,args.adamss_p)
        self.TrainSubsp_indx = TrainSubsp_indx 
        self.seg_results=seg_results 
        resW= torch.zeros((FoDs_size['num_layers'], FoDs_size['ch'], FoDs_size['d_model'], FoDs_size['d_k']),device=self.device) 
        self.AAs = []
        self.BBs = [] 
        TTsize=newWeights.shape
        GGs=torch.zeros(TTsize, device=self.device)
        for ii in range(FoDs_size['num_layers']):
            #print(ii)
            AAtemp=[]
            BBtemp=[]
            indx_ii=TrainSubsp_indx[ii]
            seg_result=seg_results[ii]
            WWr=newWeights[ii,0,:,:] 
            ZZ=ZTensor[ii,:,:]
            for jj in range(self.KK[ii]):
                #print(jj)
                indx=indx_ii[jj]
                Wr=WWr[seg_result[indx],:] 
                Z=ZZ[seg_result[indx],:]
                Z=Z[:,seg_result[indx]] 
                U, S, VT = matrixSVD(Z,device) 
                num_ii_jj = min((S > 0.1 * S[0]).sum().item(), args.adamss_ri)
                AA=U[:,:num_ii_jj]
                AAtemp.append(AA) 
                BB=torch.diag(S[:num_ii_jj])@VT[:num_ii_jj,:]
                BBtemp.append(BB)
                Z0=AA@BB 
                GGs[ii,0,seg_result[indx],:]=Z0.T@Wr 
    
            self.AAs.append(AAtemp) 
            self.BBs.append(BBtemp) 
        resW=weightTensor-torch.tensor(GGs) 
        self.resW=weightTensor.detach()
        #self.dtype=torch.float32
        print("=========Replacing the layers====")
        self.layer = multi_heads   
        self.MultiHeadAttention(FoDs_size)
        print("=========Finish the initialization====")
   
        
        
    def get_tensor(self, multi_heads, FoDs_size):
        weights_tensor = torch.zeros((FoDs_size['num_layers'], FoDs_size['ch'], FoDs_size['d_model'], FoDs_size['d_k']))
        for i, layer in enumerate(multi_heads):
            ii=0
            for qkv in self.qkv_name: 
                qkv_layer =  self.get_nested_attr_iter(layer, qkv)  
                if qkv_layer is not None:
                    if i ==0:
                        self.dtype=getattr(qkv_layer, qkv).weight.detach().dtype
                        #print(self.dtype)
                    weights_tensor[i, ii, :, :] =getattr(qkv_layer, qkv).weight.detach().to(torch.float32)#.cpu().to(self.dtype).numpy() 
                    ii=ii+1
                else:
                    print(f"Layer {i} and its descendant does NOT have attribute {qkv}")
        return weights_tensor
  
    def get_tensor_bias(self, multi_heads, FoDs_size):
        weights_tensor = np.zeros((FoDs_size['num_layers'], FoDs_size['ch'], FoDs_size['d_model'], FoDs_size['d_k']+1))
        for i, layer in enumerate(multi_heads):
            ii=0
            for qkv in self.qkv_name: 
                qkv_layer =  self.get_nested_attr_iter(layer, qkv)  
                if qkv_layer is not None and i <FoDs_size['num_layers']:
                    if i ==0:
                        self.dtype=getattr(qkv_layer, qkv).weight.detach().dtype
                    weight=getattr(qkv_layer, qkv).weight.detach().to(torch.float32) 
                    bias = getattr(qkv_layer, qkv).bias.detach().to(torch.float32)  if getattr(qkv_layer, qkv).bias is not None else torch.zeros(FoDs_size['d_model'])
                    #weights_with_bias = np.hstack((weight, bias[:, None]))
                    weights_with_bias = torch.cat((weight, bias.unsqueeze(1)), dim=1)
                    weights_tensor[i, ii, :, :] = weights_with_bias
                    ii=ii+1
                else:
                    print(f"Layer {i} and its descendant does NOT have attribute {qkv}")
        return weights_tensor
     

    def get_nested_attr_iter(self,model, objname): 
        for name, module in model.named_children(): 
            if name==objname:
                if self.FoDs_size['d_model']==module.weight.detach().cpu().to(torch.float32).numpy().shape[0]:
                   if self.FoDs_size['d_k']==module.weight.detach().cpu().to(torch.float32).numpy().shape[1]:
                     return model
                   else: 
                       return None 
                else: 
                    return None 
            else:
                result=self.get_nested_attr_iter(module,objname)
                if result is not None:
                    return result 
        return None

 
    def replace_qkv(self, module,name, resW,A,B,newA,newB,KK,TrainSubsp_indx,seg_results):
        child=getattr(module, name)
        if self.with_bias is True:
            setattr(module, name, adamss_layer_bias(resW,A,B,newA,newB,KK,TrainSubsp_indx,seg_results,self.dtype)) 
        else:
            setattr(module, name, adamss_layer(resW,A,B,newA,newB,KK,TrainSubsp_indx,seg_results,torch.Tensor(child.bias).to(self.device),self.dtype)) 

    def add_tensors(self,multi_heads_fa,name):
        W_tensor = self.adamss_Tensor 
        setattr(multi_heads_fa, name, self.adamss_Tensor) 
       
    
    def MultiHeadAttention(self, FoDs_size):
  
        for i, layer in enumerate(self.layer): 
            ii=0
            for qkv in self.qkv_name:  
                qkv_layer = self.get_nested_attr_iter(layer, qkv)
                #ii=0 
                if qkv_layer is not None and i <FoDs_size['num_layers']:
                    
                    #print(i)
                    self.replace_qkv(qkv_layer,qkv,self.resW[i,0,:,:],self.AAs[i],self.BBs[i],self.newA[i,0,:,:],self.newB[i,0,:,:],self.KK[i],self.TrainSubsp_indx[i],self.seg_results[i])     
                    ii=ii+1
                else: 
                    print(f"Layer {i} and its descendant does NOT have attribute {qkv}")
  
 
        

    
   
 
