 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from typing import Optional, List 

class SubspacesAllocator(object):
    def __init__(self, tt,target_KK, init_warmup:int, final_warmup:int, mask_interval:int,
                 beta1:float, beta2:float, total_step:Optional[int]=None):
        super(SubspacesAllocator, self).__init__()
   
        self.target_KK = target_KK 
        self.initial_warmup = init_warmup 
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval 
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step
        self.total_KK=None
        self.curr_KK=None
        self.tt=tt 
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}  

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)
        
    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step 
        assert (self.total_step>self.final_warmup and self.final_warmup>self.initial_warmup)

  

    def schedule_threshold(self,total_KK, step:int):
        # Global budget schedule
        mask_ind = False 
        target_KK = self.target_KK 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step < initial_warmup: 
            # Initial warmup 
            curr_KK = total_KK
            #is_collect=False
            mask_ind = False 
        elif step > final_warmup: 
            # Final fine-tuning 
            curr_KK = self.target_KK
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
            #is_collect=False
        else: 
            # Budget decreasing 
            #is_collect=False
            mul_coeff = 1-(step-initial_warmup)/(final_warmup-initial_warmup)
            # print('mul_coeff')
            # print(mul_coeff)
            curr_KK = target_KK + (total_KK-target_KK)*(mul_coeff**self.tt)
            curr_KK = int(curr_KK)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_KK, mask_ind 

    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lorrasa_" in n and p.grad is not None: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach() #magnitude of the gradient-weight product
                    # Update sensitivity (9)
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty (10)
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty (11)
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self,   ipt_AB):
   
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt =  ipt_AB.view(-1) 
        return sum_ipt 


    def compute_total_KK(self, model):
    
        total_KK=0
        combine_dict = {}  
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lorrasa_A" in n and p.grad is not None:  
                name_mat = n.replace("lorrasa_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [0.0]
                    total_KK=total_KK+1
            if "lorrasa_B" in n and p.grad is not None:  
                name_mat = n.replace("lorrasa_B", "%s") 
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [0.0]
                    total_KK=total_KK+1
        
  
        return total_KK

    def mask_to_target(self, model):
    
        is_dict = {}
        combine_dict = {}  
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lorrasa_A" in n and p.grad is not None: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score)
                name_mat = n.replace("lorrasa_A", "%s") 
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt] 
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lorrasa_B" in n and p.grad is not None: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score) 
                name_mat = n.replace("lorrasa_B", "%s") 
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt] 
                else:
                    combine_dict[name_mat].append(comb_ipt)  

        # Combine the importance scores
       
        all_is = []
        for name_mat in combine_dict:
              
            #ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = combine_dict[name_mat][0]+combine_dict[name_mat][1]#self._combine_ipt(ipt_AB) 
            is_dict[name_mat] = sum_ipt 
            all_is.append(sum_ipt.view(-1)) 

    
        if all_is==[] or torch.cat(all_is).size(0)<=self.curr_KK:
            return None, model 
        mask_threshold = -torch.kthvalue(-torch.cat(all_is), self.curr_KK)[0].item() 
 
   
 
        with torch.no_grad(): 
            for n,p in model.named_parameters():
                 if "lorrasa" in n and p.grad is not None:
                     if "lorrasa_A" in n:
                         name_mat = n.replace("lorrasa_A", "%s")
                     else:
                         name_mat = n.replace("lorrasa_B", "%s")
                         
                     if is_dict[name_mat]>mask_threshold:
                         p.requires_grad = True
                         #p.grad =None
                         #.detach()
                     else:
                         p.requires_grad = False
                         p.grad =None 

        return mask_threshold, model 


    def update_and_mask(self, model,  global_step):

 
 
         
        if global_step<=self.final_warmup and self.initial_warmup<=global_step:
          
            if global_step % self.mask_interval == 0:
          
                # Update importance scores element-wise 
                self.ipt = {} 
                self.exp_avg_ipt = {}
                self.exp_avg_unc = {}
                self.cat_ipt = {}
                self.update_ipt(model)
                if self.total_KK==None:
                    self.total_KK=self.compute_total_KK(model)
                self.curr_KK, mask_ind = self.schedule_threshold(self.total_KK, global_step)
        
            else:
                mask_ind=False 
        else:
            mask_ind=False 
            
   
 
        if mask_ind:  
            mask_threshold, model = self.mask_to_target(model) 
            from LorRAv2.utils import print_trainable_parameters 
            print_trainable_parameters(model) 
        else:
            mask_threshold = None  
        return mask_ind, model
 
 
