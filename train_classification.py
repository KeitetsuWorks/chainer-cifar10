#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## @file        train_classification.py
## @brief       Trainer Class for Classification CNN Training
## @author      Keitetsu
## @date        2020/05/22
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os

import random
import numpy as np
import cupy as cp

import chainer
import chainer.links as L
from chainer.training import extensions

import chainercv
from chainercv.visualizations import vis_image
from matplotlib import pyplot as plt

import cifar10_dataset
import net


class Transform():

    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, in_data):
        img, label = in_data
        self._debug_imshow(img)

        # Color augmentation
        img = chainercv.transforms.pca_lighting(img, 25.5)
        img = np.clip(img, 0, 255)
        self._debug_imshow(img)

        # Random horizontal flipping
        img = chainercv.transforms.random_flip(img, x_random=True)
        self._debug_imshow(img)

        # ランダムにキャンバスを拡張し，画像を配置
        img = chainercv.transforms.random_expand(img, max_ratio=1.5)
        self._debug_imshow(img)

        # 指定したサイズの画像を，ランダムな位置からクロップ
        img = chainercv.transforms.random_crop(img, (32, 32))
        self._debug_imshow(img)

        img /= 255.

        return img, label

    def _debug_imshow(self, img):
        if self.debug:
            vis_image(img)
            plt.show()


class TrainClassificationTask():

    def __init__(
        self,
        classes,
        train_dataset, test_dataset,
        gpu,
        model,
        batch_size, n_epoch, alpha, weight_decay,
        snapshot_interval, print_interval, output_dir
    ):
        # ラベルを読込み
        self.classes = classes

        # データセットを読込み
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # モデルを読込み
        self.model = model

        # GPUを使用する場合は，モデルをGPUにコピーする
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

        # Optimizerのセットアップ
        print("setting optimizer...: alpha=%e" % (alpha))
        self.optimizer = chainer.optimizers.Adam(alpha=alpha)
        self.optimizer.setup(self.model)
        if weight_decay:
            print("setting optimizer...: weight_decay=%e" % (weight_decay))
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        # イテレーションの設定
        print("setting iterator, updater and trainer...")
        self.train_iter = chainer.iterators.SerialIterator(
            self.train_dataset,
            batch_size,
            repeat=True,
            shuffle=True
        )
        self.test_iter = chainer.iterators.SerialIterator(
            self.test_dataset,
            batch_size,
            repeat=False,
            shuffle=False
        )
        self.updater = chainer.training.StandardUpdater(self.train_iter, self.optimizer, device=self.gpu)
        self.output_dir = output_dir
        self.trainer = chainer.training.Trainer(self.updater, (n_epoch, 'epoch'), out=self.output_dir)

        # 検証用データセットで評価する
        self.trainer.extend(extensions.Evaluator(self.test_iter, self.model, device=self.gpu))
        # 学習途中でスナップショットを取得する
        self.trainer.extend(
            extensions.snapshot(filename='snapshot_iter_{.updater.epoch}'),
            trigger=(snapshot_interval, 'epoch')
        )
        # 学習途中でモデルのスナップショットを取得する
        self.trainer.extend(
            extensions.snapshot_object(
                self.model,
                filename='snapshot_model_{.updater.epoch}',
                savefun=chainer.serializers.save_hdf5
            ),
            trigger=(snapshot_interval, 'epoch')
        )
        # グラフを取得する
        self.trainer.extend(
            extensions.PlotReport(
                [
                    'main/loss',
                    'validation/main/loss'
                ],
                x_key='epoch',
                file_name='loss.png',
                marker=""
            )
        )
        self.trainer.extend(
            extensions.PlotReport(
                [
                    'main/accuracy',
                    'validation/main/accuracy'
                ],
                x_key='epoch',
                file_name='accuracy.png',
                marker=""
            )
        )
        # ログを取得する
        self.trainer.extend(extensions.LogReport())
        # 学習と検証の状況を表示する
        self.trainer.extend(
            extensions.PrintReport(
                [
                    'epoch',
                    'main/loss',
                    'validation/main/loss',
                    'main/accuracy',
                    'validation/main/accuracy',
                    'elapsed_time'
                ]
            ),
            trigger=(print_interval, 'epoch')
        )
        # プログレスバーを表示する
        self.trainer.extend(extensions.ProgressBar())

    def run(self):
        print("starting training...")
        self.trainer.run()

        print("saving model...")
        model_file_path = os.path.join(self.output_dir, 'net.model')
        chainer.serializers.save_hdf5(model_file_path, model)
        
        print("training is complete")


def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer (CuPy) random seed
    cp.random.seed(seed)


if __name__ == '__main__':
    # 乱数のシードを設定
    set_random_seed(0)

    # データセットを読込み
    dataset = cifar10_dataset.CIFAR10Dataset(
        train=True
    )

    # データセットを学習用と検証用に分割する
    # 90%を学習用とする
    dataset_split_rate = int(len(dataset) * 0.9)
    train_dataset, test_dataset = chainer.datasets.split_dataset_random(
        dataset,
        dataset_split_rate,
        seed=0
    )

    # Transformクラスを使用して，データセットの水増しを行う
    train_transform_dataset = chainer.datasets.TransformDataset(
        train_dataset,
        Transform(debug=False)
    )
    test_transform_dataset = chainer.datasets.TransformDataset(
        test_dataset,
        Transform(debug=False)
    )

    # モデルを読込み
    # * L.Classifierでは予測した値とラベルとの誤差を計算する．
    #   デフォルトではsoftmax_cross_entropy
    print("loading model...")
    model = L.Classifier(net.Net(len(dataset.classes)))

    train_task = TrainClassificationTask(
        classes=dataset.classes,
        train_dataset=train_transform_dataset,
        test_dataset=test_transform_dataset,
        gpu=0,
        model=model,
        batch_size=100,
        n_epoch=200,
        alpha=0.00005,
        weight_decay=0.0001,
        snapshot_interval=10,
        print_interval=1,
        output_dir='./logs'
    )

    train_task.run()
