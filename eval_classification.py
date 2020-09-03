#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## @file        eval_classification.py
## @brief       Evaluation Class for Classification CNN
## @author      Keitetsu
## @date        2020/05/24
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

import matplotlib.pyplot as plt
import seaborn as sns

import cifar10_dataset
import net


class EvalClassificationTask:

    def __init__(
        self,
        classes,
        dataset,
        gpu,
        model,
        output_dir
    ):
        # ラベルを読込み
        self.classes = classes

        # データセットを読込み
        self.dataset = dataset

        # モデルを読込み
        self.model = model

        # GPUを使用する場合は，モデルをGPUにコピーする
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

        self.output_dir = output_dir

    def run(self):
        n_tests = len(self.dataset)
        print("number of test data: %d" % (n_tests))
        
        n_classes = len(self.classes)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        n_acc = 0

        print("starting evaluation...")
        for i in range(0, n_tests):
            # 1つのアノテーションデータに対して推論
            label, result_label = self.eval_example(i)

            # 推論結果をconfusion_matrixに反映
            confusion_matrix[label, result_label] += 1

            # 正解数をカウント
            if label == result_label:
                n_acc += 1
        
        # 結果を表示
        print("confusion matrix")
        print(confusion_matrix)
        self.plot_confusion_matrix(confusion_matrix)
        print("# corrests: %d" % (n_acc))
        print("accuracy = %f" % (float(n_acc) / n_tests))
        print("evaluation is complete")

    def eval_example(self, i):
        # データセットからデータを取得
        img, label = self.dataset[i]
        img = chainer.Variable(img[None, ...])

        # GPUを使用する場合は，画像をGPUにコピーする
        if self.gpu >= 0:
            img.to_gpu()

        # 推論を実行
        x = self.model.predictor(img)
        result = F.argmax(x)

        if self.gpu >= 0:
            result.to_cpu()
        result_label = result.data

        return label, result_label

    def plot_confusion_matrix(self, cm):
        sns.heatmap(
            cm,
            vmin=0,
            cmap='Blues',
            annot=True,
            fmt="d",
            square=True,
            xticklabels=self.dataset.classes,
            yticklabels=self.dataset.classes,
        )
        plt.ylabel("true label")
        plt.xlabel("predictecd label")
        cm_file_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_file_path, bbox_inches='tight')


if __name__ == '__main__':
    # データセットを読込み
    dataset = cifar10_dataset.CIFAR10Dataset(
        train=False
    )

    # モデルを読込み
    print("loading model...")
    chainer.config.train = False
    model = L.Classifier(net.Net(len(dataset.classes)))
    chainer.serializers.load_hdf5('./logs/net.model', model)

    # 評価を実行する
    eval_task = EvalClassificationTask(
        classes=dataset.classes,
        dataset=dataset,
        gpu=0,
        model=model,
        output_dir='./logs'
    )

    eval_task.run()
