# microplasticDBbaseLine

This repository privide a Pytorch based tool fo

## The dataset

Holo_microplastic is a dataset providing..

## Installation

```
pip install -r requirements.txt
```

## Run a banckmark

The banchmark parameter can be set-up by means a configuration file.
You can fine an example in `./config/customConf.yaml` that will be the default one in case a specific file will not be provided.
Configuration file also include the list of network you intend to test. Currently the available Networks are:
- Alexnet (alexnet)
- VGG 11 (vgg11)
- ResNet18 (resNet18)
- Mobilenetwork V2 (mobilenet_v2)

In order to run the experiment:

```
python banckmark.py --device gpu --name newtest --config ./config/costuomConf.yaml --dataset <basepath>/Microplastiche/images --gt <basepath>/gt.csv
```

- *--name* It is the name of the current experiment. A subfolder with the experiment name will be created in *tests* folder and will contains all the results. (mandatory parameter)
- *--dataset* It is the path of the dataset (mandatory parameter)
- *--gt* It ts the path of the csv ground truth.
- *--config* It is the path of the configuration file.
- *--device* default is *cpu*, but *gpu* (Nvidia Cuda) or *mps* (M1 gpu) can be selected.

## Generate report
`report.py` allows to generate a report of the run banckmarks. It will returns ROC curves plots and accuracy of each network.

```
python report.py --name newtest 
```

## Inference

Banckmark automatically save the models for each fold. Such models can be easly used for inference.
```
 python predict.py --model ./trainedModels/resNet18.pt --image ./imges/img.bmp
```

## Statistics
The script `statistic.py` returns the statistics in terms of images dimension for a given subset provided with the groundtruth.


