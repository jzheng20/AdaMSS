import torch.nn as nn
#from .TensorTools.TensorSVD import *
from .utilized import * 
from .lrr import *


 

 

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
            Q, _,_ =torch.svd_lowrank((newA[seg_result[TrainSubsp_indx[i]],:]).T@A[i], q=B[i][0], niter=2, M=None)
            setattr(self,f'adamss_A_{i}',nn.Parameter(Q.T.to(dtype))) 
            setattr(self, f'adamss_B_{i}', nn.Parameter(torch.zeros(B[i][1],B[i][0]).to(dtype)))
        #Only need B size
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
        x7=torch.zeros_like(x6) 
        x7[:,:,self.newindex]=x6
         
        return  x1+x7

 
 
 
class replacing_with_AdaMSS(nn.Module):
    def __init__(self,args, index,  multi_heads, FoDs_size,qkv_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(replacing_with_AdaMSS, self).__init__()
        
        FoDs_size['num_layers']=1
 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype=None
 
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
        weightTensor=weightTensor.to(self.device) 
     
        VVT, UU=slicePCA(weightTensor,args.adamss_R,self.device,torch.float32)
 
        self.newA=UU
        self.newB=VVT 
        ZTensor=torch.zeros(weightTensor.shape[0],weightTensor.shape[2],weightTensor.shape[2],device=self.device)  
        indx,self.KK=clustering_Z(UU[0,0,:,:],args.adamss_K,10)  
        ZTensor=ZTensor.to(self.device)
        self.seg_results=seg_locations(weightTensor.shape[0],indx) 
        self.TrainSubsp_indx=get_trainable_subspaces_all(FoDs_size['num_layers'],self.KK)  
        self.Initialization_MD(args,FoDs_size,ZTensor,UU[0,0,:,:]) 
        self.resW=weightTensor.detach()  
        self.layer = multi_heads   
        self.MultiHeadAttention(FoDs_size)
 
   
    def Initialization_MD(self,args,FoDs_size,ZTensor,U):
        self.AAs = []
        self.BBs = [] 
      
 
        for ii in range(FoDs_size['num_layers']):
            #print(ii)
            AAtemp=[]
            BBtemp=[]
            indx_ii=self.TrainSubsp_indx[ii]
            seg_result=self.seg_results[ii]
  
            ZZ=ZTensor[ii,:,:]
            for jj in range(self.KK[ii]): 
                indx=indx_ii[jj] 
                num_ii_jj = min(len(seg_result[indx]), args.adamss_ri) 
                AAtemp.append(U[seg_result[indx],:]) 
                BB=np.array([num_ii_jj, len(seg_result[indx])])
                BBtemp.append(BB)
          
    
            self.AAs.append(AAtemp) 
            self.BBs.append(BBtemp)   
        
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
                    print(f"Layer {i} does NOT have attribute {qkv}")
        return weights_tensor
   


    def get_tensor_bias(self, multi_heads, FoDs_size):
        weights_tensor = torch.zeros((FoDs_size['num_layers'], FoDs_size['ch'], FoDs_size['d_model'], FoDs_size['d_k']+1))
        layer=multi_heads
        for qkv in self.qkv_name:
            qkv_layer =  self.get_nested_attr_iter(layer, qkv)
            if qkv_layer is not None:
                self.dtype=getattr(qkv_layer, qkv).weight.detach().dtype
                weight=getattr(qkv_layer, qkv).weight.detach().to(torch.float32)
                bias = getattr(qkv_layer, qkv).bias.detach().to(torch.float32)  if getattr(qkv_layer, qkv).bias is not None else torch.zeros(FoDs_size['d_model']) 
                weights_with_bias = torch.cat((weight, bias.unsqueeze(1)), dim=1)
                weights_tensor[0, 0, :, :] = weights_with_bias
   
            else:
                print(f"Layer {self.index} and its descendant does NOT have attribute {qkv}")
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
        for qkv in self.qkv_name:
            qkv_layer = self.get_nested_attr_iter(self.layer, qkv)
            if qkv_layer is not None: 
                self.replace_qkv(qkv_layer,qkv,self.resW[0,0,:,:],self.AAs[0],self.BBs[0],self.newA[0,0,:,:],self.newB[0,0,:,:],self.KK[0],self.TrainSubsp_indx[0],self.seg_results[0])     
 
            else:
                 print(f"Layer {self.index} does NOT have attribute {qkv}")
  
 
        

    
   
 