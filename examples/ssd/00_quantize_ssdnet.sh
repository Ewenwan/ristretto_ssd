#!/usr/bin/env sh

./build/tools/ristretto quantize \
        --model=models/VGGNet/VOC0712/SSD_300x300/train.prototxt \
        --weights=models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
        --model_quantized=models/ssd/quantized.prototxt \
        --trimming_mode=dynamic_fixed_point --gpu=0 --iterations=2000 \
        --error_margin=3
