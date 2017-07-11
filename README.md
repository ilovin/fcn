## train
```shell
chmod 755 train.sh
./train.sh
```
## test
```shell
chmod 755 test.sh
./test.sh
```

## result
![fcn](result/result.png)
## run
```shell
python ./fcn/train_net.py --cfg ./experiments/cfgs/fcn_nlpr.yml --network FCN_32s  --iters 300000 --restore 0 --pre_train=./data/VGGnet_imagenet.npy
python ./lib/utils/converckpt2npy.py   to get `32s.npy`
python ./fcn/train_net.py --cfg ./experiments/cfgs/fcn_nlpr.yml --network FCN_16s  --iters 300000 --restore 0 --pre_train=./data/32s.npy
```
then train FCN_8s

## setup
### Requirements: software
1. tensorflow ,Python packages: `cython`, `python-opencv`, `easydict` (recommend to install: [Anaconda](https://www.continuum.io/downloads))

2. link the preprocessing modeule
    `ln -sf tensorflow/models/slim/preprocessing ./lib/fcn/utils/preprocessing`
