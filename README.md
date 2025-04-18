# [DA6401 Assignment 2](https://wandb.ai/sivasankar1234/DA6401/reports/DA6401-Assignment-2--VmlldzoxMjAyNjgyNw)



This repository contains the implementation for **DA6401 Assignment 2**, a computer vision project focused on classifying images from the iNaturalist dataset using a custom Convolutional Neural Network (CNN) and a fine-tuned ResNet50 model. The project includes data preprocessing, model training, hyperparameter tuning via Weights & Biases (WandB) sweeps, and visualization of predictions.

- **Assignment Report**: [WandB Report](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment2/reports/DA6401-Assignment-2--VmlldzoxMjA5OTc1MQ)
- **GitHub Repository**: [aayushmanda/da6401_assignment2](https://github.com/aayushmanda/da6401_assignment2)
- **WandB Sweep Links**:
  - [Sweep Homepage (Scratch with Parallel Plot)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment2/sweeps/letqkeos?nw=nwuserda24s016)
  - [Sweep Homepage (Scratch)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/da6401-assignment2?nw=nwuserda24s016)
  - [Sweep Homepage (Fine-tuning ResNet50)](https://wandb.ai/da24s016-indian-institute-of-technology-madras/resnet50-pytorch-tuning/sweeps/im2xbayo?nw=nwuserda24s016)


---
Please install the dataset before running any of the .py files
```bash
 wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip
 unzip nature_12K.zip
```

## Dataset
The project uses the [**iNaturalist 12K dataset**](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). Download and extract it as follows:

```
da6401_assignment2/
├── PartA/                    # Files for custom CNN implementation
│   ├── config.yaml           # Configuration for dataset paths and hyperparameters
│   ├── visualization.py      # Utility for plotting prediction grid
│   ├── da6401_A.ipynb        # Jupyter notebook for training and evaluation
│   ├── train.py              # Script for training the custom CNN
│   ├── model.py              # Custom CNN model definition (FlexibleCNN)
│
├── PartB/                    # Files for ResNet50 fine-tuning
│   ├── config.yaml           # Configuration for dataset paths and hyperparameters
│   ├── train.py              # Script for fine-tuning ResNet50
│   ├── sweep.ipynb           # Notebook for hyperparameter sweeps with WandB
│   ├── model.py              # ResNet50 model setup and fine-tuning logic
│   ├── da6401_B.ipynb        # Jupyter notebook for fine-tuning and evaluation
│
├── requirements.txt          # Python dependencies for both parts
├── README.md                 # Project overview and setup instructions
```