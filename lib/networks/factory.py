# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}
from .FCN_train import FCN_train
from .FCN_test import FCN_test
from .VGGnet_32s import VGGnet_32s
from .VGGnet_16s import VGGnet_16s
from .VGGnet_8s import VGGnet_8s
def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'FCN':
        if name.split('_')[1] == 'train':
            return FCN_train()
        elif name.split('_')[1] == 'test':
            return FCN_test()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))

    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == '32s':
            return VGGnet_32s()
        elif name.split('_')[1] == '16s':
            return VGGnet_16s()
        elif name.split('_')[1] == '8s':
            return VGGnet_8s()
        else:
            raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return list(__sets.keys())

