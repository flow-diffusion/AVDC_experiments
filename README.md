# AVDC experiments

The official codebase for running the experiments described in the AVDC paper. 
code base for training video policies [here](https://github.com/flow-diffusion/AVDC).

[Learning to Act from Actionless Video through Dense Correspondences](https://flow-diffusion.github.io/AVDC.pdf)  
[Po-Chen Ko](https://pochen-ko.github.io/),
[Jiayuan Mao](https://jiayuanm.com/),
[Yilun Du](https://yilundu.github.io/),
[Shao-Hua Sun](https://shaohua0116.github.io/),
[Joshua B. Tenenbaum](https://cocosci.mit.edu/josh)  
[website](https://flow-diffusion.github.io/) | [paper](https://flow-diffusion.github.io/AVDC.pdf) | [arXiv](https://arxiv.org/abs/2310.08576)

```bib
@article{Ko2023Learning,
  title={{Learning to Act from Actionless Video through Dense Correspondences}},
  author={Ko, Po-Chen and Mao, Jiayuan and Du, Yilun and Sun, Shao-Hua and Tenenbaum, Joshua B},
  journal={arXiv:2310.08576},
  year={2023},
}
```

## Getting started

We recommend to create a new environment with pytorch installed using conda.  

```bash  
conda create -n avdc_exp python=3.9
conda activate avdc_exp
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```  

Next, clone the repository and install the requirements  

```bash
git clone https://github.com/flow-diffusion/AVDC_experiments
cd AVDC_experiments
pip install -r requirements.txt
```

## Download the Checkpoints

We provide the checkpoints used in our main experiments. You can download them using `download.sh`, for example:

```bash
bash download.sh metaworld
# bash download.sh metaworld_DA
# bash download.sh metaworld_ithor 
```

## Running the Experiments

First, cd into the `experiment` directory. 

```bash
cd experiment
```

### Meta-World

To run the full AVDC on Meta-World, run the following command:

```bash
# make sure you have the checkpoint ../ckpts/metaworld/model-24.pt
bash run_metaworld.sh
```

We have provided also provided another checkpoint trained with simple random-shift data augmentation. Specifically we first center cropped the image to 160x160 from the original 320x240 image and then random-crop an 128x128 image from it. 

```bash
# make sure you have the checkpoint ../ckpts/metaworld_DA/model-24.pt
bash run_metaworld_DA.sh
```

### iTHOR

To run the full AVDC on iTHOR, run the following command:

```bash
# make sure you have the checkpoint ../ckpts/ithor/model-24.pt
bash run_ithor.sh
```

## Acknowledgements

This codebase is modified from the following repositories:  
[unimatch](https://github.com/autonomousvision/unimatch)
[imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)  
[guided-diffusion](https://github.com/openai/guided-diffusion)  


