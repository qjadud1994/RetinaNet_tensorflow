# RetinaNet_tensorflow
For easier and more readable tensorflow codes

## How to use
- For Trainig (recommend to use the default parameters)
```
python tfrecord/tfrecord_VOC.py
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
- For Testing (recommend to use the default parameters)
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Todo list:
- [x] multi-gpu code
- [x] Training visualize using Tensorboard
- [x] validation output image visualization using Tensorboard
- [x] Choose BatchNorm model or GroupNorm model
- [x] Choose Trainable BatchNorm(not working!) or Freeze BatchNorm 
- [x] (BatchNorm mode) Get Imagenet pre-trained weights from [pytorch-resnet50.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- [x] (GroupNorm mode) Get Imagenet pre-trained weights from [resnet50_groupnorm32](http://www.cs.unc.edu/~cyfu/resnet50_groupnorm32.tar)
- [ ] add evaluation (AP) code
- [ ] change upsample function for 600x600 input
- [ ] Training/Validation Error ( % value)
- [ ] add augmentation ( + random crop & rotate)
- [ ] tf.train.batch -> tf.train.shuffle_batch

## Description
|       File         |Description                                                   |
|----------------|--------------------------------------------------|
|train.py  |  Train RetinaNet            |
|test.py |  Inference RetinaNet            |
|tfrecord/tfrecord_VOC. py | Make VOC tfrecord |
|Detector/layers. py | layer functions used in RetinaNet  |
|Detector/RetinaNet. py | Define RetinaNet |

## Environment

- os : Ubuntu 16.04.4 LTS <br>
- GPU : Tesla P40 (24GB) <br>
- Python : 3.6.6 <br>
- pytorch : 0.3.1
- CUDA, CUDNN : 9.0, 7.0.5 <br>
