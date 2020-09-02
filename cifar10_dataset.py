# -*- coding: utf-8 -*-

##
## @file        cifar10_dataset.py
## @brief       CIFAR10 Dataset Class
## @author      Keitetsu
## @date        2020/08/28
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import chainer


class CIFAR10Dataset(chainer.dataset.DatasetMixin):

    def __init__(
        self,
        train=True
    ):
        self.classes = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        )

        train_dataset, test_dataset = chainer.datasets.get_cifar10(scale=255.)
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        # 画像とラベルを取得
        x, t = self.dataset[i]

        return x, t
