import argparse, time, logging, os, tempfile

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true',
                    help='use image record iter for data input. default is false.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=32, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--params', type=str, default='',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--normalize-input', action='store_true',
                    help='use normalized input for imagenet data. default is false.')
opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(opt)

batch_size = opt.batch_size
classes = 1000

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

model_name = opt.model

kwargs = {'ctx': context, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if(len(opt.params)<=1):
  kwargs['pretrained']=True
  opt.normalize_input=True

net = get_model(model_name, **kwargs)
logging.info(net)

def convert_params(arg_dict, dtype='float32'):
  for k in arg_dict:
    arg_dict[k] = arg_dict[k].astype(dtype)
  return

if(len(opt.params)>1):
  print('try to load model param from ',opt.params)
  defult_tmp_dir = tempfile._get_default_tempdir()
  temp_name = os.path.join(defult_tmp_dir,next(tempfile._get_candidate_names()))
  arg_dict = nd.load(opt.params)
  convert_params(arg_dict)
  nd.save(temp_name, arg_dict)
  try:
    net.collect_params().load(temp_name,ctx=context)
  except:
    net.load_parameters(temp_name,ctx=context)
  os.remove(temp_name)
net.cast(opt.dtype)

# Two functions for reading data from record file or raw images
def get_data_rec(rec_val, rec_val_idx, batch_size, num_workers):
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    if(opt.normalize_input):
        print('using noramlized input.')
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]
    else:
        mean_rgb=[0,0,0]
        std_rgb=[1,1,1]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = 256,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    if(opt.normalize_input):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = None

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return val_data, batch_fn

if opt.use_rec:
    val_data, batch_fn = get_data_rec(opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)
else:
    val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

def test(ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def log_test(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    tic = time.time()
    err_top1_val, err_top5_val = test(ctx, val_data)
    logging.info('time cost: %f'%(time.time()-tic))
    logging.info('validation: err-top1=%f err-top5=%f'%(err_top1_val, err_top5_val))

def main():
    log_test(context)

if __name__ == '__main__':
    main()
