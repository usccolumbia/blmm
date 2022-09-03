# Crystal-Composition-Transformer
This repository contains the code and datasets for the paper:

[**Crystal Transformer: Self-learning neural language model for Generative and Tinkering Design of Materials**](https://arxiv.org/pdf/2204.11953.pdf)  
*Lai Wei, Qinyang Li, Yuqi Song, Edirisuriya M. D. Siriwardane, Stanislav Stefanov, and Jianjun Hu*

by <a href="http://mleg.cse.sc.edu" target="_blank">Machine Learning and Evolution Laboratory</a>, University of South Carolina.

### Python Dependencies
Install `Pytorch` from [Pytorch web](https://pytorch.org/get-started/previous-versions/) given your python & cuda version

The code is based on the [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework. It has been tested in `PyTorch 1.6.0`, `PyTorch Lightning 1.0.7`

The version of Pymatgen package: `pip install pymatgen==2021.2.16`

### Datasets for training Crystal Composition Transformer

|       | ICSD-mix | OQMD-mix | MP-mix | ICSD-pure | OQMD-pure | MP-pure |
|-------|----------|----------|--------|-----------|-----------|---------|
| Total | 52317    | 363182   | 89121  | 39431     | 216540    | 63703   |
| Train | 50755    | 345022   | 84664  | 37459     | 205713    | 60517   |
| Valid | 1336     | 9080     | 2228   | 986       | 5413      | 1593    |
| Test  | 1336     | 9080     | 2228   | 986       | 5413      | 1593    |

All above datasets can be downloaded from [Figshare](https://figshare.com/articles/dataset/BLMM_dataset/20489964)

### Acknowledgements

We use the blank language model from [https://github.com/Varal7/blank_language_model/edit/release/README.md](https://github.com/Varal7/blank_language_model)

### How to train the model with Crystal Composition Transformer dataset

#### Download Data
Download datasets from the above link, then unzip it under `BLMM_dataset.zip` folder.
After the above, the directory should be:
```
Crystal Composition Transformer
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
  author={Wei, Lai and Li, Qinyang and Song, Yuqi and Stefanov, Stanislav and Siriwardane, Edirisuriya and Chen, Fanglin and Hu, Jianjun},
  journal={arXiv preprint arXiv:2204.11953},
  year={2022}
}
```
