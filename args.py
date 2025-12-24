import argparse
import os
 
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--dataset", type=str, default="mrpc")
parser.add_argument("--task", type=str, default="mrpc")
parser.add_argument("--bs", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=1) 
parser.add_argument("--head_lr", type=float, default=5e-3)
parser.add_argument("--fft_lr", type=float, default=1e-1)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_step", type=float, default=0.06)
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--scale", type=float, default=300.)
parser.add_argument("--width", type=float, default=200.)
parser.add_argument("--fc", type=float, default=1.)
parser.add_argument("--share_entry", action= "store_true")
parser.add_argument("--use_abs", action= "store_true")
parser.add_argument("--set_bias", action= "store_true")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--entry_seed", type=int, default=2024)


## For ViT computer vision tasks
parser.add_argument("--model-name-or-path", type=str,
                    required=True,
                    default="google/vit-base-patch16-224-in21k")
parser.add_argument("--dataset-name", type=str,
                    required=True,
                    choices=[
                        "flowers",
                        "pets",
                        "dtd",
                        "food",
                        "resisc",
                        "eurosat",
                        "cars",
                        "fgvc",
                        "cifar10",
                        "cifar100",
                    ]) 
parser.add_argument("--mode", type=str, 
                    required=True, 
                    choices=["adamss",
                             "loretta",
                             "lora",
                             "pissa", 
                             "pretrained", 
                             "head", 
                             "full"])  



parser.add_argument("--multi_heads", type=str, default="vit.encoder.layer")
parser.add_argument("--multi_heads_fa", type=str, default="vit.encoder")
#parser.add_argument("--lorrasa_EK", type=str, default="False")
parser.add_argument("--adamss_EK", type=str, default="True")
parser.add_argument("--trainable_subspaces", type=str, default="all")


parser.add_argument("--adamss_R", type=int, default=500)
parser.add_argument("--adamss_K", type=int, default=5)
parser.add_argument("--adamss_ri", type=int, default=1)
parser.add_argument("--adamss_p", type=float, default=0.2)

parser.add_argument("--target_KK", type=int, default=50)
parser.add_argument("--target_ll", type=int, default=10000)
parser.add_argument("--init_warmup", type=int, default=50)
parser.add_argument("--final_warmup", type=int, default=100)
parser.add_argument("--mask_interval", type=int, default=100)
parser.add_argument("--beta1", type=float, default=0.85)
parser.add_argument("--beta2", type=float, default=0.85)
parser.add_argument("--MODE_SA", type=str, default="True")
parser.add_argument("--file_root", type=str, default="lrr_pkg")
parser.add_argument("--tt", type=float, default=0.01)
 

parser.add_argument("--n_trial", type=int, default=1) 

parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--cache-dir", type=str, default=os.path.join(os.getenv("HOME"), ".cache"))
parser.add_argument("--data-local-dir", type=str, default=None)
 

args = parser.parse_args()

def get_args():
    return parser.parse_args()
