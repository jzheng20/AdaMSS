import torch.nn as nn
 
from .utilized import * 
from tqdm import tqdm



orth_ini=True



if orth_ini:
    from .Replacing_orth import * 
else:
    from .Replacing_random import * 
 
 
def Loading(args,  model,  FoDs_size,layers_name,device):
  
    KKK=[]
    
    total = sum(FoDs_size[i]["num_layers"] for i in range(len(layers_name)))
    multi_heads = get_nested_attr(model, args.multi_heads)
    with tqdm(total=total, desc="Initializing All Layers") as pbar:
        for i  in  range(len(layers_name)):
            layernum=FoDs_size[i]["num_layers"]
            for ii in range(layernum): 
                HH=replacing_with_AdaMSS(args, ii, multi_heads[ii], FoDs_size[i],layers_name[i],device)
                KKK.append(HH.KK)
                pbar.update(1)
        
    print("=========Finish the initialization====")
    print(model) 
        
    return model, KKK
         

  
        
    
    
 
