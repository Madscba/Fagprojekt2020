#!/bin/sh
dos2unix cnn_lr1.sh
dos2unix cnn_lr2.sh
dos2unix cnn_lr3.sh
dos2unix cnn_lr4.sh
dos2unix cnn_lr5.sh

bsub < cnn_lr1.sh
bsub < cnn_lr2.sh
bsub < cnn_lr3.sh
bsub < cnn_lr4.sh
bsub < cnn_lr5.sh


