# mxnet-quantized
mxnet quantized models and related works

# Usage:
python score_imagenet.py --num-gpus 1 --use-rec --model mobilenet0.5 --params models/0.3443-imagenet-qmobilenet0.5-best.params --normalize-input

# model-zoo


|model| float top-1 \*| quantized top-1 |
|--|--|--|
|resnet18_v1|70.93||
|[mobilenet0.5 int8](./models/0.3443-imagenet-qmobilenet0.5-best.params)|65.20| 65.57 |

> \* [GluonCV Model zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html)
