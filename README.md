### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  https://github.com/chaitu2289/ristretto_ssd
  cd ristretto_ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  make runtest -j8
  # If you have multiple GPUs installed in your machine, make runtest might fail. If so, try following:
  export CUDA_VISIBLE_DEVICES=0; make runtest -j8
  # If you have error: "Check failed: error == cudaSuccess (10 vs. 0)  invalid device ordinal",
  # first make sure you have the specified GPUs, or try following if you have multiple GPUs:
  unset CUDA_VISIBLE_DEVICES
  ```

### Preparation
1. Download [trained PASCAL VOC models(07 + 12)](www.cs.unc.edu/%7Ewliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz).
```Shell
   cd $HOME
   tar -zxvf models_VGGNet_VOC0712_SSD_300x300.tar.gz
```
   This will create the following files
```Shell
   models/VGGNet/VOC0712/SSD_300x300/
   models/VGGNet/VOC0712/SSD_300x300/test.prototxt
   models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt
   models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
   models/VGGNet/VOC0712/SSD_300x300/solver.prototxt
   models/VGGNet/VOC0712/SSD_300x300/train.prototxt
   models/VGGNet/VOC0712/SSD_300x300/ssd_pascal.py
   models/VGGNet/VOC0712/SSD_300x300/score_ssd_pascal.py
```

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

4. Create a directory `ssd` in `$CAFFE_ROOT/models`
   ```Shell
   cd $CAFFE_ROOT
   mkdir models/ssd
   ```

5. Run the following command to create the fixed point version of the ssd model

```Shell
   ./build/tools/ristretto quantize \
        --model=models/VGGNet/VOC0712/SSD_300x300/train.prototxt \
        --weights=models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
        --model_quantized=models/ssd/quantized.prototxt \
        --trimming_mode=dynamic_fixed_point --gpu=0 --iterations=2000 \
        --error_margin=3
```



## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }



