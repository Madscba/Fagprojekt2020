#!/bin/sh
dos2unix get_balance.sh
dos2unix cnn_lr5_new.sh
dos2unix cnn_lr4_new.sh
dos2unix cnn_lr4.sh
dos2unix cnn_lr5.sh
bsub < get_balance.sh
bsub < cnn_lr5_new.sh
bsub < cnn_lr4_new.sh
bsub < cnn_lr4.sh
bsub < cnn_lr5.sh