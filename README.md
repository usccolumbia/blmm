# BLMM  Crystal-Composition-Transformer
This repository contains the code and datasets for the paper:

[**Crystal Transformer: Self-learning neural language model for Generative and Tinkering Design of Materials**](https://arxiv.org/pdf/2204.11953.pdf)  
*Lai Wei, Qinyang Li, Yuqi Song, Stanislav Stefanov,Rongzhi Dong, Nihang Fu, Edirisuriya M. D. Siriwardane, Fanglin Chen and Jianjun Hu*

by <a href="http://mleg.cse.sc.edu" target="_blank">Machine Learning and Evolution Laboratory</a>, University of South Carolina.

### Running environment set up

The BLM language model code we used is from [here](https://github.com/Varal7/blank_language_model), which is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. It has been tested in PyTorch 1.6.0, PyTorch Lightning 1.0.7

Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) based on your python & cuda version

```
conda create -n blm
conda activate blm
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pytorch-lightning=1.0.7

or for Nvidia 3090
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.0.7
```

Install Pymatgen package: 
```
pip install pymatgen==2021.2.16
```
### Datasets for training Crystal Composition Transformer

|       | ICSD-mix | OQMD-mix | MP-mix | ICSD-pure | OQMD-pure | MP-pure |
|-------|----------|----------|--------|-----------|-----------|---------|
| Total | 52317    | 363182   | 89121  | 39431     | 216540    | 63703   |
| Train | 50755    | 345022   | 84664  | 37459     | 205713    | 60517   |
| Valid | 1336     | 9080     | 2228   | 986       | 5413      | 1593    |
| Test  | 1336     | 9080     | 2228   | 986       | 5413      | 1593    |

All above datasets and a pretrained model files can be downloaded from [Figshare](https://figshare.com/articles/dataset/BLMM_dataset/20489964)

### Source code:

We use the blank language model from [https://github.com/Varal7/blank_language_model](https://github.com/Varal7/blank_language_model). Please dowload the code following the link.

```
git clone https://github.com/Varal7/blank_language_model.git
cd blank_language_model

```


### How to generate new materials composition using our pretrained model:

Download the pretrained model files blmm_model.zip from [Figshare](https://figshare.com/articles/dataset/BLMM_dataset/20489964) and put it inside the source code folder blank_language_model.

unzip the BLMM_model.zip file

```
cd blank_language_model
cp blmm-model/hparams.yaml ./
cp blmm-model/vocab.txt ./

python test.py --checkpoint blmm-model/icsd-mix-model.ckpt --sample 1000 --decode sample --output sample.txt
or
python test.py --checkpoint blmm-model/icsd-mix-model_epoch2249.ckpt --sample 1000 --decode sample --output sample.txt
```


### How to train your own model with Crystal Composition Transformer datasets

#### Download Data
Download datasets from the above link, then unzip it under `BLMM_dataset` folder.
After the above, the directory should be:

blank_language_model
   ├── BLMM_dataset
       ├── mix_dataset
           ├── icsd_train.txt
           ├── icsd_valid.txt
           ├── icsd_test.txt
           ├── oqmd_train.txt
           ├── oqmd_valid.txt
           ├── oqmd_test.txt
           ├── mp_train.txt
           ├── mp_valid.txt
           ├── mp_test.txt
       ├── pure_dataset
           ├── icsd_train.txt
           ├── icsd_valid.txt
           ├── icsd_test.txt
           ├── oqmd_train.txt
           ├── oqmd_valid.txt
           ├── oqmd_test.txt
           ├── mp_train.txt
           ├── mp_valid.txt
           ├── mp_test.txt
   ├── blmm_model
           ├── hparams.yaml
           ├── icsd-mix-model.ckpt
           ├── vocab.txt
   └── README.md
```



#### How to train the model
An example is to train a BLMM model on the icsd_mix dataset. 
```
python train.py --train BLMM_dataset/mix_dataset/icsd_train.txt --valid BLMM_dataset/mix_dataset/icsd_valid.txt --root_dir checkpoints/icsd_mix/blm/ \
--vocab_size 130 --max_len 210 --model_type blm --share_emb_prj_weight
```
The training for other models is similar to icsd_mix dataset.

#### How to generate new hypothesis materials using the trained models
For all of the following, replace `epoch\=???.ckpt` with the checkpoint saved in training.

An example to generate hypothesis materials using the trained icsd_mix model.
```
python test.py --checkpoint checkpoints/icsd_mix/blm/lightning_logs/version_0/checkpoints/epoch\=???.ckpt \
--sample 1000 --decode sample --output sample.txt
```

### Citation

If you use our work, please cite:

```bibtex
@article{wei2022crystal,
  title={Crystal Transformer: Self-learning neural language model for Generative and Tinkering Design of Materials},
  author={Wei, Lai and Li, Qinyang and Song, Yuqi and Stefanov, Stanislav, rongzhi dong, nihang fu, and Siriwardane, Edirisuriya and Chen, Fanglin and Hu, Jianjun},
  journal={arXiv preprint arXiv:2204.11953},
  year={2022}
}
```


### Acknowledgement

Our code is derived from the [Blank Language Model](https://github.com/Varal7/blank_language_model) for text generation. See [Paper](https://arxiv.org/abs/2002.03079)
