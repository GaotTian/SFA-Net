# SFA-Net: A SAM-guided Focused Attention Network for Multimodal Remote Sensing Image Matching

Tian Gao, Chaozhen Lan*, Wenjun Huang, Sheng Wang, Zijun Wei, Fushan Yao, Longhao Wang

## Framework
![img](https://github.com/GaotTian/SFA-Net/blob/main/framework.png)
## Dependencies
The python environment is 3.7 and requires the following third-party libraries to be installed：
```
time
opencv
segment_anything
typing
numpy
torch
copy
math
os
```
where segment_anything needs to be installed separately
Download ```segment-anything-main.zip``` locally, unzip it and ```python setup.py install``` it. Or ```pip install git+https://github.com/facebookresearch/segment-anything.git```

## Dataset
1、Megadepth.
2、OS-dataset. 
 
## Testing and visualisation
The weights have been uploaded to the folder ./weights
```shell script
python main.py
```
