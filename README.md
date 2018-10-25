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
- [ ] add evaluation (AP) code
- [ ] change upsample function for 600 input
- [ ] validation infernece image visualization using Tensorboard
- [ ] Training/Validation Error ( % value)
- [ ] add augmentation ( + random crop & rotate)
- [ ] tf.train.batch -> tf.train.shuffle_batch

## Description
|       File         |Description                                                   |
|----------------|--------------------------------------------------|
|train.py  |  Train RetinaNet            |
|test.py |  Inference RetinaNet            |
|tfrecord/tfrecord_VOC. py | Make VOC tfrecord |
|Detector/layers. py | layer function used in RetinaNet  |
|Detector/RetinaNet. py | Define RetinaNet |

## Environment

- os : Ubuntu 16.04.4 LTS <br>
- GPU : Tesla P40 (24GB) <br>
- Python : 3.6.6 <br>
- pytorch : 0.3.1
- CUDA, CUDNN : 9.0, 7.0.5 <br>
