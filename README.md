# On the Benefit of Adversarial Learning for Monocular Depth Estimation
This is the repository for our CVIU work **On the Benefit of Adversarial Learning for Monocular Depth Estimation**. \
[arXiv](https://arxiv.org/pdf/1910.13340.pdf) \
[CVIU](https://doi.org/10.1016/j.cviu.2019.102848)

Two works have served as baselines for this work:  
**Unsupervised Monocular Depth Estimation with Left-Right Consistency**  
[arXiv](https://arxiv.org/abs/1609.03677)

**Unsupervised Adversarial Depth Estimation using Cycled Generative Networks** 
[arXiv](https://arxiv.org/pdf/1807.10915.pdf)  


This repository implements the basic training and evaluation code, to prevent clutter.

## Dependencies
A [requirements](requirements.txt) file is available to retrieve all dependencies. Create a new python environment and install using:
```shell
pip install -r requirements.txt
``` 

## Training
Models can be trained by specifying your data directory, a model name and any architecture.
```shell
python main.py --data_dir data/ --model_name [MODEL_NAME] --architecture wgan
```
Resume training is possible by filling in the resume flag with the path to the saved model:  
```shell
python main.py --data_dir data/ --model_name [MODEL_NAME] --architecture wgan --resume saved_models/[MODEL_NAME]/model_best.pth.tar
```
There are many, many options for training the models. Have a look at the [options](options/) with three python files containing options for training, testing and evaluation.

## Testing  
To test change the `--mode` flag to `test`, the network will output the disparities in the [output](output) folder. 
```shell
python main.py --data_dir data/ --model_name [MODEL_NAME] --mode test
```

## Evaluation of Depth
Run the following script to run any evaluation, given that a disparities file is present in [output](output):
```shell
python evaluate.py --data_dir data/ --predicted_disp_path output/disparities_[DATASET]_[MODEL_NAME].npy  
```

## Data
This work has been trained on rectified stereo pairs. For this two datasets have been used: KITTI and CityScapes.
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
In this work the split of **eigen** is used to train and test model. This set contains 22600 training images, 888 validation imagesn and 697 test images.  
In the [filenames](utils/filenames) folder there are lists that detail which images correspond to which set. All data can be downloaded by running:
```shell
wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/
```

### [CityScapes](https://www.cityscapes-dataset.com)
To access data of the CityScapes dataset, one has to register an account and then request special access to the ground truth disparities.  
When this data is retrieved the following directories should be put in the [data](data/) folder:  
cs_camera/ with all camera parameters.  
cs_disparity/ with all ground truth disparities.  
cs_leftImg8bit/ with all left images.  
cs_rightImg8bit/ with all right images.

## Results
Results are available upon request.

## References
A few repositories were the inspiration for this work. These are:

[Unsupervised Monocular Depth Estimation with Left-Right Consistency](https://github.com/mrharicot/monodepth/blob/master/readme.md)  
[Unsupervised Adversarial Depth Estimation using Cycled Generative Networks](https://github.com/andrea-pilzer/unsup-stereo-depthGAN/blob/master/README.md)  
[Club AI's Pytorch Implementation of MonoDepth](https://github.com/ClubAI/MonoDepth-PyTorch)  
[Cycle GAN and Pix2Pix in Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Citation
If this work was useful for your research, please consider citing:
```
@article{groenendijk2020benefit,
  title={On the benefit of adversarial training for monocular depth estimation},
  author={Groenendijk, Rick and Karaoglu, Sezer and Gevers, Theo and Mensink, Thomas},
  journal={Computer Vision and Image Understanding},
  volume={190},
  pages={102848},
  year={2020},
  publisher={Elsevier}
}
```

