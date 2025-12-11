import torch 
import numpy as np
#import matplotlib.pyplot as plt
from scipy.fftpack import dct 

 

def get_nested_attr(obj, attr):
    attributes = attr.split('.')
    for attribute in attributes:
        if hasattr(obj, attribute):
            obj = getattr(obj, attribute)
        else:
            return None
    return obj

def set_extra_trainable(model, extra_tune: list[str]):
    for n, p in model.named_parameters():
        if any([t in n for t in extra_tune]):
            p.requires_grad = True

# def get_nested_attr_iter(model, objname):
#     for name, module in model.named_children():
#         if name==objname:
#             return model
#         else:
#             result=get_nested_attr_iter(module,objname)
#             if result is not None:
#                 return result 
#     return None


    
def get_trainable_para_num(model):
    lst=[]
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            lst.append(param.nelement())
    print(f"trainable parameters number: {sum(lst)}")
    
def check_requires_grad(model,prefix):
    for name, param in model.named_parameters():
        #print(name)
        if name.startswith(prefix):
            print('Trainable parameters:' + name)
            param.requires_grad=True
        else:
            print('Untrainable parameters:' + name)
            param.requires_grad=False
    get_trainable_para_num(model)

def print_requires_grad(model):
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad) 
    get_trainable_para_num(model)

def replace_attention_layers(model, obj1,obj2):
    print(model.named_children())
    for name, module in model.named_children(): 
        if name==obj1:  
            setattr(model, name, obj2) 
        else:
            replace_attention_layers(module, obj1, obj2)
     
