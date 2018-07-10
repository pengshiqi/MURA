# [Bone X-Ray Deep Learning Competition](https://stanfordmlgroup.github.io/competitions/mura/)

[![](https://img.shields.io/badge/language-python-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/framework-pytorch-blue.svg)](https://pytorch.org/)

Path on server: /DATA4_DB3/data/sqpeng/Projects/MURA

## LOGs

🌀 ***2018-06-11 13:02***

Accuracy=66.31%

🌀 ***2018-06-12 08:09***

Accuracy=71.60%

🌀 ***2018-06-13 01:45***

Accuracy=83.95%



Average:

|    model     |   ELBOW   |  FINGER   |  FOREARM  |   HAND    |  HUMERUS  | SHOULDER  |   WRIST   |  AVERAGE  |
| :----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Baseline** | **0.710** | **0.389** | **0.737** | **0.851** | **0.600** | **0.729** | **0.931** | **0.705** |
| DenseNet169  |   0.746   |   0.582   |   0.711   |   0.540   |   0.778   |   0.576   |   0.701   |   0.662   |
|  ResNet152   |   0.733   |   0.569   |   0.650   |   0.480   |   0.793   |   0.576   |   0.710   |   0.644   |
|    VGG16     |   0.632   |   0.592   |   0.650   |   0.495   |   0.689   |   0.566   |   0.742   |   0.624   |
|   Ensemble   |   0.760   |   0.569   |   0.681   |   0.480   |   0.807   |   0.586   |   0.729   |   0.659   |



Min:

|    model     |   ELBOW   |  FINGER   |  FOREARM  |   HAND    |  HUMERUS  | SHOULDER  |   WRIST   |  AVERAGE  |
| :----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **Baseline** | **0.710** | **0.389** | **0.737** | **0.851** | **0.600** | **0.729** | **0.931** | **0.705** |
| DenseNet169  |   0.764   |   0.608   |   0.683   |   0.545   |   0.763   |   0.579   |   0.770   |   0.673   |
|  ResNet152   |   0.735   |   0.665   |   0.696   |   0.533   |   0.733   |   0.557   |   0.714   |   0.662   |
|    VGG16     |   0.614   |   0.561   |   0.682   |   0.507   |   0.674   |   0.518   |   0.712   |   0.610   |
|   Ensemble   |   0.734   |   0.594   |   0.742   |   0.555   |   0.719   |   0.578   |   0.750   |   0.667   |



1. 数据预处理。
2. 考虑 Collaborative Learning 。
3. 考虑 各个部位共享一部分网络，然后单独训练。



## Dataset

Path: /DATA4_DB3/data/public/MURA-v1.1

Description:

* 7 folders: XR_ELBOW, XR_FINGER, XR_FOREARM, XR_HAND, XR_HUMERUS, XR_SHOULDER, XR_WRIST

* Training data: 36808 images, 13457 cases

* Validation data: 3197 images, 1199 cases

* Training Data Distribution

|  body parts  | positive | negative | total  |
| :----------: | :------: | :------: | :---:  |
| ELBOW        |  2006    |  2925    |  4931  |
| FINGER       |  1968    |  3138    |  5106  |
| FOREARM      |  661     |  1164    |  1825  |
| HAND         |  1484    |  4059    |  5543  |
| HUMERUS      |  599     |  673     |  1272  |
| SHOULDER     |  4168    |  4211    |  8379  |
| WRIST        |  3987    |  5765    |  9752  |
| ALL          |  14873   |  21935   |  36808 |

* Test Data Distribution

|  body parts  | positive | negative | total  |
| :----------: | :------: | :------: | :---:  |
| ELBOW        |  230     |  235     |  465   |
| FINGER       |  247     |  214     |  461   |
| FOREARM      |  151     |  150     |  301   |
| HAND         |  189     |  271     |  460   |
| HUMERUS      |  140     |  148     |  288   |
| SHOULDER     |  278     |  285     |  563   |
| WRIST        |  295     |  364     |  659   |
| ALL          |  1530    |  1667    |  3197  |

Images in training set are grey-scale images, i.e., there is only one channel.

The sizes of these images are various. Minimum width is 89, and minimum height is 132. Maximum width is 512, and maximum height is 512.


## References

* [PyTorch Book](https://github.com/chenyuntc/pytorch-book)

