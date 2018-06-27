# [Bone X-Ray Deep Learning Competition](https://stanfordmlgroup.github.io/competitions/mura/)

[![](https://img.shields.io/badge/language-python-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/framework-pytorch-blue.svg)](https://pytorch.org/)

Path on server: /DB/rhome/sqpeng/PycharmProjects/MURA

## LOGs

🌀 ***2018-06-11 13:02***

Accuracy=66.31%

🌀 ***2018-06-12 08:09***

Accuracy=71.60%

🌀 ***2018-06-13 01:45***

Accuracy=83.95%

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
