# DeepLearning_GTSRB
A Deep Learning project focused on classifying traffic signs using the GTSRB dataset.

## Prerequisites
**MATLAB R2024b** (or above) with **Deep Learning Toolbox**.

## Usage
Download the GTSRB dataset from this link:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

You should get an archive named "archive.zip"; if not, rename it as such and place it into the src/ subdirectory.

Then, launch the driver script **DLA_tsr.m** and you should be good to go.

You also need the models **resnet18** and **mobilenetv2** installed from the MATLAB app **Deep Network Designer**; if you don't have them when launching **DLA_tsr.m**, a message will be displayed on console, containing a link that allows you to install them effortlessly.
