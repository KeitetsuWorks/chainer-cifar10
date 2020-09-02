# -*- coding: utf-8 -*-

##
## @file        net.py
## @brief       Convolutional Neural Network
## @author      Keitetsu
## @date        2020/08/28
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import chainer
import chainer.functions as F
import chainer.links as L


class Net(chainer.Chain):
    def __init__(self, n_class):
        initializer = chainer.initializers.HeNormal()
        super(Net, self).__init__(
            conv1=L.Convolution2D(None, 32, 3, stride=1, pad=1, nobias=True, initialW=initializer),
            conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1, nobias=True, initialW=initializer),
            conv3=L.Convolution2D(None, 64, 3, stride=1, pad=1, nobias=True, initialW=initializer),
            fc1=L.Linear(None, 512, nobias=True, initialW=initializer),
            fc2=L.Linear(None, n_class, nobias=True, initialW=initializer)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, ratio=0.25)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, ratio=0.25)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, ratio=0.25)
        h = self.fc2(h)
        return h
