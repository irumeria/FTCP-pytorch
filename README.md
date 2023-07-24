## Pytorch Implement of Fourier-Transformed Crystal Properties (FTCP)

This is a simple Pytorch implementation of the paper [An invertible crystallographic representation for general inverse design of inorganic crystals with targeted properties](https://www.cell.com/matter/pdf/S2590-2385(21)00625-1.pdf). The representation-building code is reserved, and the model is rewrited in Pytorch.

### Prerequisites
```bash
pip install -r requirements.txt
```
Pytorch install command is recommended on [Pytorch official website](https://pytorch.org/get-started/locally/), if you want a exactly GPU version of it.

### Usage
```bash
python train.py # training
python infer.py # sampling
```

### Config
You can change the hyperparameters in the variable `params` from `train.py` file. 
```python
params = {  'num_conv_layers' : 3,
            'embedding_dim' : 128,
            'kernel1_size' : 5,
            'kernel2_size' : 3,
            'kernel3_size' : 3,
            'strides1' : 2,
            'strides2' : 2,
            'strides3' : 1,
            'latent_dimensions' : 256,
            'batch_size' : 256,
            'epochs' : 300,
            'dim_pp' : 128,
            'property_predictor':True,
            'learning_rate' : 1e-4,
            'max_elms': 5,
            'max_sites': 40,
            'device': device,
        }
```
The parameters of infer.py will be loaded from the checkpoint.

### Dataset
The example dataset in `data/example.csv` of is from the [Material Project](https://next-gen.materialsproject.org/)

### Other Implements

+ Original (in tensorlfow 1.15): https://github.com/PV-Lab/FTCP

