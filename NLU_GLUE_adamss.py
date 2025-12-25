import torch

def print_memory_usage():
    # 已分配的显存
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    # 当前保留的显存
    reserved = torch.cuda.memory_reserved() / 1024**2   # 转换为 MB
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")


import numpy as np
from args_glue import *
import os
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
#from peft import (
#    get_peft_config,
#    get_peft_model,
#    get_peft_model_state_dict,
 #   set_peft_model_state_dict,
#    LoraConfig, 
#    PeftType,
#    PrefixTuningConfig,
#    PromptEncoderConfig, 
#)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from adamss_pkg import utilized


def set_tM_trainable(model,module_name):
    for name, module in model.named_children():
        if name in module_name:
            for name0, param in module.named_parameters():
                if "tM" in name0:
                    param.requires_grad=True
                
            #return model
        else:
            set_bias_trainable(module,module_name) 
    return None

args = get_args() 
print(args.weight_decay)
#seeds=np.array([44444]) #,11111,22222,33333,,44444
seeds=np.array([args.seed])  
for seed in seeds:
    args.seed=seed
    print(args.seed) 

    torch.manual_seed(args.seed)
    task = args.task
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    num_labels = 2
    if task == "stsb":
        num_labels = 1
    
 

    def log(*pargs):
        path_log = './logs_glue/' + task + '/' + args.model_name_or_path.split("-")[1] + '/bs' + str(args.bs) + 'maxlen' + str(args.max_length) + 'f_lr' + str(args.fft_lr)+ 'h_lr' + str(args.head_lr) + \
          'num' + str(args.n_frequency) + 'scale' + str(args.scale) + 'seed' + str(args.seed) + '.txt'
        print(path_log)
        with open(path_log, mode = 'a+') as w:
            w.write(" ".join(["{}".format(t) for t in pargs]))
            w.write("\n")

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else: 
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
 
    datasets = load_dataset(args.dataset_path+"datasets_nlu/glue/glue/"+task)  
    #metric=load_metric("glue",task)  
    from evaluate import load
    metric = load("lrr_pkg/evaluate/metrics/glue/glue.py", task) 

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        if task == 'sst2' or task == 'cola':
            outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qnli':
            outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qqp':
            outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
        else:
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
        return outputs

    if task == 'sst2' or task == 'cola':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
        )
    elif task == 'qnli':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question", "sentence"],
        )
    elif task == 'qqp':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question1", "question2"],
        )
    else:
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")


    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs)

 
    for fft_lr in np.array([1e-3]): #1e-1,1e-2,1e-3
        for head_lr in np.array([5e-2]):#5e-2,0.0005,5e-5
            for weight_decay in np.array([0]):#5e-2, 5e-3,5e-4,0
                #args.adamss_K=10
                # args.fft_lr=fft_lr#1e-3
                # args.head_lr=head_lr
                # args.weight_decay= weight_decay
                
                print("hahahaha")
                print(args.head_lr)
                print(args.fft_lr)
                print(args.weight_decay)
                print(args.adamss_K)
                print(args.adamss_ri)
                print(args.adamss_R)
                model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
                from adamss_pkg import AdaMSSModel,utilized
                for param in model.parameters():
                    param.requires_grad = False 
                args.transform_type='HOSVD' 
                qkv_name=[['query'],['value']]  
                if "base" in args.model_name_or_path:
                    target_size=np.array([12,768,768])
                elif "large" in args.model_name_or_path:
                    target_size=np.array([24,1024,1024])
                # if "base" in args.model_name_or_path:
                #        FoDs_size = [{"num_layers": 12,"ch": len(qkv_name[0]),"d_model": 768,"d_k": 768},
                #                     {"num_layers": 12,"ch": len(qkv_name[0]),"d_model": 768,"d_k": 768}] 
                # else:
                #        FoDs_size = [{"num_layers": 24,"ch": len(qkv_name[0]),"d_model": 1024,"d_k": 1024},
                #                     {"num_layers": 24,"ch": len(qkv_name[0]),"d_model": 1024,"d_k": 1024}] 

                FoDs_size = [{"num_layers": target_size[0],"ch": len(qkv_name[0]),"d_model": target_size[1],"d_k": target_size[2]},
                        {"num_layers": target_size[0],"ch": len(qkv_name[0]),"d_model": target_size[1],"d_k": target_size[2]}] 
                args.ll=12*768*2
                # r=100
                print(args)
                args.adamss_p=0.2 
                # args.target_KK=10
                model, args.KK=AdaMSSModel.Loading(args, model, FoDs_size,qkv_name,device)
                print(args.KK)
    
                #for param in model.parameters():
                     #   param.requires_grad = False 
                utilized.set_extra_trainable(model, ["classifier"]) 
                utilized.print_requires_grad(model)
                #set_tM_trainable(model,["query","value"])
                #utilized.print_requires_grad(model)
                #print(model) 
                head_param = list(map(id, model.classifier.parameters()))

                others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 
        
                optimizer = AdamW([{"params": model.classifier.parameters(), "lr": args.head_lr},{"params": others_param,                         "lr":args.fft_lr}],weight_decay=args.weight_decay)
                # Instantiate scheduler
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
                    num_training_steps=(len(train_dataloader) * args.num_epochs),
                    )
                from adamss_pkg.asa import SubspacesAllocator
                if args.MODE_SA == "True":
                    subspaces_allocator =  SubspacesAllocator(
                        tt=args.tt, 
                        target_KK=args.target_KK,
                        init_warmup=args.init_warmup,
                        final_warmup=args.final_warmup,
                        mask_interval=args.mask_interval,
                        beta1=args.beta1,
                        beta2=args.beta2,
                        )
                else:
                    subspaces_allocator=None
                torch.cuda.empty_cache()
                acc_list = []
                model.to(device)
                max_steps=  args.num_epochs * len(train_dataloader)
                print("max_steps")
                print(max_steps)
                total_KK=int(sum(args.KK[0]))
                from adamss_pkg.utils import print_trainable_parameters
                print_trainable_parameters(model)
                #print(dir())
                #print(globals())
                import time
                start = time.time()
                for epoch in range(args.num_epochs):
                    model.train()
                    for step, batch in enumerate(tqdm(train_dataloader)):
                        if subspaces_allocator is not None:
                            subspaces_allocator.set_total_step(args.num_epochs)
                        batch.to(device)
                        outputs = model(**batch)
                        loss = outputs.loss  
                        loss.backward()
                        if step == 0:
                            print_memory_usage()
                        optimizer.step()
                        #if step == 0: 
                      
                        lr_scheduler.step()
                        if subspaces_allocator is not None and step == 0:
                             mask_threshold, model =  subspaces_allocator.update_and_mask(model, epoch)
        
                        if step == 0: #step % 100 == 0:
                            torch.cuda.empty_cache() 
                            print(f"Max allocated memory for Net1: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                            torch.cuda.reset_peak_memory_stats()
                            
                             # for param in model.parameters():
                             #     param.requires_grad = False
                        optimizer.zero_grad()
                        #torch.cuda.empty_cache()
                        # head_param = list(map(id, model.classifier.parameters()))
                        # others_param = filter(lambda p: id(p) not in head_param and p.requires_grad, model.parameters())  
                        # optimizer = AdamW([{"params": model.classifier.parameters(), "lr": args.head_lr},{"params": others_param,                         "lr":args.fft_lr}],weight_decay=args.weight_decay)
                        

                    model.eval()
                    for step, batch in enumerate(tqdm(eval_dataloader)):
                        batch.to(device)
                        with torch.no_grad():
                            outputs = model(**batch)
                        if task == "stsb":
                            predictions = outputs.logits
                        else:
                            predictions = outputs.logits.argmax(dim=-1)
                        predictions, references = predictions, batch["labels"] 
                        metric.add_batch(
                            predictions=predictions,
                            references=references,
                            )

                    eval_metric = metric.compute()
                    if task == "stsb":
                        acc_list.append(eval_metric['pearson']) 
                        print(f"epoch {epoch}:", eval_metric, '\033[32m,current_best_pearson:\033[0m',max(acc_list),'train_loss:',loss)
                    elif task == 'cola':
                        acc_list.append(eval_metric['matthews_correlation'])
                        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss) 
                    else:
                        acc_list.append(eval_metric['accuracy'])
                        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list),'train_loss:',loss) 
          
                end = time.time()
                print(f"运行时间: {end - start:.4f} 秒")
                    

        
        

