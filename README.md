# ATSS-EfficientDet-PyTorch

ATSS-EfficientDet ([ATSS](https://arxiv.org/pdf/1912.02424.pdf) built on top of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf)) ```outperforms``` the original EfficientDet.

This repository is folked from [mmdetection](https://github.com/open-mmlab/mmdetection), implemented in [PyTorch](https://pytorch.org/).

## Introduction

There are more applications using Deep Learning in Computer Vision like Image Classification, Object Detection, and Semantic Segmentation. However, Deep Learning models, for example CNN, are too slow. Therefore, more effort is paid in order to make these models running faster.

This work was activated by the [EfficientDet-family networks](https://arxiv.org/pdf/1911.09070.pdf) in Object Detection, published by Google team, which consume small GFLOPs without accuracy compensation. Besdies, [ATSS](https://arxiv.org/pdf/1912.02424.pdf) is a sampling method for Anchor-based architectures (e.g., RetinaNet), aiming at boosting the detection accuracy with only single anchor at each location.

This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and tries to:
  - reimplement EfficientDet in PyTorch
  - build ATSS on top of EfficientDet
  - train ATSS-EfficientDet with small versions (D0, D1)

<p align="center">
  <img src="demo/atss_effdet_d0/demo.jpg" height="250" alt="accessibility text">
  <img src="demo/atss_effdet_d0/img1.jpg" height="250" alt="accessibility text">
  <br>
  <em>Demo images predicted by ATSS-EfficientDet-D0</em>
</p>

See more demo images in [demo/atss_effdet_d0](demo/atss_effdet_d0)


## Comparison

|         Model        | Params (M) | FLOPs (G) | COCO-val mAP | Official COCO-val mAP |
|:--------------------:|:----------:|:---------:|:------------:|:---------------------:|
| ATSS-EfficientDet-D0 |    3.83    |    2.32   |     33.8     |          33.5         |


## Installation

Please refer to the official guildeline of [mmdetection](https://github.com/open-mmlab/mmdetection) in [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.


## References

* Two papers:
  - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
  - [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/pdf/1912.02424.pdf)

* Three repositories:
  - [mmdetection](https://github.com/open-mmlab/mmdetection)
  - [automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet)
  - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


## Citation

```
@article{atssefficientdet,
  title   = {ATSS-EfficientDet: ATSS built on top of EfficientDet},
  author  = {Thuy Nguyen-Chinh},
  journal= {https://github.com/thuyngch/ATSS-EfficientDet-PyTorch},
  year={2020}
}
```
