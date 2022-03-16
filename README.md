# Vision Transformer Module networks  (ViTMs)
![Python version support](https://img.shields.io/badge/python-3.5%20%203.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/pytorch-0.2%200.3%200.4-red.svg)

This repository contains code for replicating the experiments and visualizations from the paper:

----------------------------- ADD PAPER HERE -----------------------------

The following repositories provide the basis for the code in this repository:
- [https://github.com/facebookresearch/clevr-iep](https://github.com/facebookresearch/clevr-iep)
- [https://github.com/davidmascharka/tbd-nets](https://github.com/davidmascharka/tbd-nets)

# Training a Model
To train a model from scratch, there are a few requirements to take care of. We assume you have
already [set up PyTorch](#pytorch) and [CUDA/cuDNN](#cudacudnn) if you plan on using a GPU (which is
highly recommended).

### 1. Getting data
The CLEVR dataset is available at [its project page](http://cs.stanford.edu/people/jcjohns/clevr/).
The first step for training is to download that data.

You will also need to extract features and process the question files to produce programs before
training a model. The [instructions
here](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md#preprocessing-clevr)
provide a method for this. We recommend cloning that repository and following those instructions.

For feature extraction, use:

``` shell
python scripts/extract_features.py \
    --input_image_dir data/CLEVR_v1.0/images/train \
    --output_h5_file data/train_features.h5 \
```

For generating programs:

```bash
python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file data/train_questions.h5 \
  --output_vocab_json data/vocab.json

python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --output_h5_file data/val_questions.h5 \
  --input_vocab_json data/vocab.json
  
python scripts/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_test_questions.json \
  --output_h5_file data/test_questions.h5 \
  --input_vocab_json data/vocab.json
```

When preprocessing questions, we create a file `vocab.json` which stores the mapping between
tokens and indices for questions and programs. We create this vocabulary when preprocessing
the training questions, then reuse the same vocabulary file for the val and test questions.


After you have finished the above, you will have several HDF5 files containing the image features
and questions, and a vocabulary file. While we do provide a `DataLoader` that will work with the
HDF5 files, we personally find NumPy npy files more robust and generally more pleasant to work with,
so we default to using those.

##### a. Converting HDF5 to npy
Note that this step is completely optional. The [h5_to_np script](utils/h5_to_np.py) will produce
npy files from your HDF5 files.

Note that the largest NumPy data file (train_features.npy) is 53 gigabytes for the 14x14 feature
maps or 105 gigabytes for the 28x28 feature maps, meaning you will need a substantial amount of RAM
available on your machine to create these files. *If you do not have enough memory available, use
the HDF5 data loader instead of trying to convert these files.*

To convert your HDF5 files to npy files, invoke one of the following, depending on whether you want
to convert images to NumPy format as well:

``` shell
python h5_to_np -q /path/to/questions.h5 -f /path/to/features.h5 -i /path/to/images.h5 -d /path/to/dest/
python h5_to_np -q /path/to/questions.h5 -f /path/to/features.h5 -d /path/to/destination/
```

### 2. Training the model
The [train notebook](scripts/train_model_final_LR_Decay1.ipynb) will then walk through the training process. 
The notebook is written for a Google Colab notebook and assumes the data and files are stored in Google Drive. 
This code will need to be edited to find the relevant files and data.
```

If you prefer a different directory structure, update the data loader paths in the notebook. The
notebook will walk through training a model from this point.

# Testing a Model
Note that the testing data does not provide ground truth programs, so we will need to generate
programs from the questions for testing. We do not focus on this component of the network in our
work, so we reuse the program generator from [Johnson *et
al.*](https://github.com/facebookresearch/clevr-iep).
The [test notebook](scripts/test.ipynb) will walk through the 
process to produce a file containing the predicted test answers.

# Notes

### Python
We only recommend running the code with Python 3, having done all our development using Python
3.6. 

### PyTorch
Our development was done using PyTorch v1.5.0. For setting up PyTorch, see the [official installation instructions](https://github.com/pytorch/pytorch#installation). The specific hash that the original model from our paper was developed from is
[here](https://github.com/pytorch/pytorch/tree/d9b89a352c4ceeff24878f4f5321e16f059e98c3).

To use PyTorch <0.4, clone the repository and check out `tags/torch0.3`. For PyTorch 0.4 and above, `master` will run.

### CUDA/cuDNN
Our code is tested under CUDA 11. For setting up
CUDA, see the [NVIDIA documentation](https://developer.nvidia.com/cuda-toolkit). We recommend using
cuDNN, which is also available [from NVIDIA](https://developer.nvidia.com/cudnn).

### Operating Systems
Our development was done on Ubuntu 16.04. The code has also been tested under Arch
Linux.

### Setting up a conda environment
If you like, you can use the `environment.yml` configuration to set up a development environment if
you use `conda`. This is the environment that Binder uses to give a live notebook for the
visualizations. To create an environment using this, run

``` shell
conda env create -f environment.yml
```

The environment can then be activated with `source activate votm-env`.
