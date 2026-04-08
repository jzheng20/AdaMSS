
# AdaMSS: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning

This repository contains the official PyTorch implementation of **AdaMSS (Adaptive Multi-Subspace Approach)**, a parameter-efficient fine-tuning (PEFT) method designed to improve the expressiveness–efficiency trade-off in fine-tuning of large models.


##  🔧 Installation

```bash
git clone https://github.com/jzheng20/AdaMSS.git
cd AdaMSS
conda create -n adamss python=3.12.2
conda activate adamss
pip install -r requirements.txt
```

## 📄 Citation

If you find this work or code useful, please consider citing:

```BibTeX
@inproceedings{zheng2025adamss,
  title={AdaMSS: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning},
  author={Zheng, Jingjing and Lu, Wanglong and Dong, Yiming and Ji, Chaojie and Cao, Yankai and Lin, Zhouchen},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```




## Results on GLUE for given subspace number K=10 (hyperparameter configuration follows the AdaMSS paper), evaluated on an NVIDIA Tesla V100 (32 GB).

| Model | # Tranable Parameters | cola |mrpc |qnli |rte | sst2 | stsb | Avg.
|-------|----|----|----| ----| ----| ----| ----| ----|  
| AdaMSS_base (K=10, r_k=1) | 0.097M |0.6882 ± 0.0153  | 0.9005 ± 0.0063|0.9424 ± 0.0026|  0.8838 ± 0.0065| 0.9631 ± 0.0017|0.9181 ± 0.0021|0.8827|
| AdaMSS  (K=10, r_k=1) |  0.045M | 0.6866 ± 0.0064 |0.8985 ± 0.0037 |0.9426 ± 0.0019 |0.8744 ± 0.0077 |0.9612 ± 0.0015 |0.9178 ± 0.0019 |0.8802 | 


## 📦 PEFT Integration

Our AdaMSS has been integrated into the PEFT package!!!! Please check here for more details of peft integration: [AdaMSS-PEFT](https://github.com/huggingface/peft/tree/main/examples/adamss_finetuning)

###  Example Usage of AdaMSS

```bash
from peft import AdamssConfig, get_peft_model

# Configure AdaMSS with ASA
config = AdamssConfig(
    r=100,                          # SVD rank (full decomposition rank)
    num_subspaces=10,               # Number of subspaces (K) - initial capacity
    subspace_rank=1,                # Rank per subspace (ri) - use 1 for NLU, 3 for Vision
    target_modules=["query", "value"],  # Target attention layers
    modules_to_save=["classifier"], # Modules to train without decomposition
)
peft_model = get_peft_model(model, config)
```
