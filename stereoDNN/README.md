# Stereo DNN TensorRT inference library
The goal of this project is to enable inference for [NVIDIA Stereo DNN](https://arxiv.org/abs/1803.09719) TensorFlow models on Jetson as well as other platforms supported by NVIDIA TensorRT library. You can see the inference on KITTI dataset video demo [here](https://youtu.be/0FPQdVOYoAU)

This is a 2-step process:
1. Convert the TensorFlow model to TensorRT C++ API model. This step is performed only once for each model and can be done on any environment like user's desktop.
2. Use the TRT C++ API model in an application. Once the model is built, it can be used in any environment (e.g. Jetson) to perform inference. 

Note: TensorFlow is **not** required for the inference step. The library needs only basic components like CUDA 9.0, cuDNN 7.0 and TensorRT 3.0 so it will run as-is on Jetson with JetPack 3.2

The library implements the following TensorRT plugins:
* `conv3d`          : implementation of TensorFlow-compatible 3D convolution
* `conv3d_transpose`: implementation of TensorFlow-compatible 3D transposed convolution (aka deconvolution)
* `cost_volume`     : implementation of cost volume computation used in StereoDNN
* `softargmax`      : implementation of specific softargmax implementation used in StereoDNN
* `elu`             : implementation of ELU activation function
* `transform`       : implementation of tensor transformation required for certain operations
* `slice`           : implementation of tensor slicing required for certain operations
* `pad`             : implementation of tensor padding required for certain operations

Note that these plugins make certain assumptions that are valid in case of Stereo DNN.
`slice` and `pad` plugins implement only a tiny piece of functionality required to run the inference. 

## Models
There are several Stereo DNN models included in this packages, the following table provides brief comparison. `TF` stands for TensorFlow and `TRT` - for our implementation based on TensorRT and cuDNN. All times are in milliseconds per image, averaged over 200 images.

| Model        | Input size  | Titan Xp (TF) | Titan Xp (TRT) | Jetson TX2 (TRT) FP32 / FP16 | D1 error (%) |
| ---------    | ----------- | --------------| -------------- | ---------------------------  | ------------ |
| NVSmall      | 1025x321    |       800     |       450      |       7800 / NA              |      9.8     |
| NVTiny       |  513x161    |        75     |        40      |        360 / NA              |     11.12    |
| ResNet-18    | 1025x321    |       950     |       650      |      11000 / NA              |     3.4(*)   |
| ResNet-18 2D |  513x257    |        15     |        9       |        110 / 55              |     9.8      |

Notes:
* We could not run TensorFlow on Jetson with our models so no measurments were done in this case.
* D1 error for `NVSmall` and `NVTiny` was measured on 200 training images from [KITTI 2015 stereo benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo). This dataset was **not** used to train the models.
* `*` - measured on [KITTI 2015 stereo test set](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo). Note that this model was fine-tuned on 200 training images, so providing error on that dataset is not useful.
* FP16 is currently enabled only for `ResNet-18 2D` model.

## Converting TensorFlow model to TensorRT C++ API model
To convert TensorFlow model to TensorRT C++ API, run the `./scripts/model_builder.py` script which takes several named parameters: 
* model type
* network name to use in generated C++ code
* path to TensorFlow model
* path to resulting binary weights file 
* path to generated C++ file. 

You can also optionally specify model data type (fp32 or fp16 with fp32 being default).

Example:
```sh
cd ./scripts/
python ./model_builder.py --model_type nvsmall --net_name NVSmall1025x321 --checkpoint_path=../models/NVSmall/TensorFlow/model-inference-1025x321-0 --weights_file=../models/NVSmall/TensorRT/trt_weights.bin --cpp_file=../sample_app/nvsmall_1025x321_net.cpp --data_type fp32
```
Currently the supported model types are `nvsmall` and `resnet18`. `NVTiny` is a slight variation of `NVSmall` so it works with `nvsmall` model type. Adding new model types should be relatively easy, `./scripts/model_nvsmall.py` or `./scripts/model_resnet18.py` can provide a good starting point.

Note: TensorFlow v.1.5 or later is required. We stronly recommend using our [TensorFlow Docker container](../tools/tensorflow/docker) as it contains all necessary components required to use Stereo DNN with TensorFlow.

## Building inference code
Once the TensorRT C++ model is created, it can be used in any TensorRT-enabled application. The inference static library `nvstereo_inference` located at `./lib/` contains imlpementation of TensorRT plugins requried to run Stereo DNN. A sample application located at `./sample_app/` provides example of library usage. To build library, sample application and tests, run the following commands:

```sh
# Build debug:
mkdir build
cd ./build/
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
# Build release:
cd ..
mkdir build_rel
cd ./build_rel/
cmake -DCMAKE_BUILD_TYPE=Release ..
```

If you get CMake error that `GTest` is not found, do the following:
```sh
cd /usr/src/gtest
cmake CMakeLists.txt
make
```
and then try building the library again (you may need to use `sudo` depending on your environment).

The build will place binary files in `./bin/` directory.

It's a good idea to run the tests first to make sure everything is working as expected:
```sh
./bin/nvstereo_tests_debug ./tests/data
```
All tests should pass (obviously). We recommend running debug version first to make sure all asserts in the code are enabled.

To run the sample application:
```sh
./bin/nvstereo_sample_app_debug nvsmall 513 161 ./models/NVTiny/TensorRT/trt_weights.bin ./sample_app/data/img_left.png ./sample_app/data/img_right.png ./bin/disp.bin
```
The app takes 8 parameters:
* model type (`nvsmall` or `resnet18`)
* dimensions of the image (width and height - must be equal to dimensions of network input)
* path to weights file created by model builder script
* 2 images, left and right (e.g. PNG files)
* path to output file, the app will create 2 files: binary and PNG
* [optional] data type (fp32 or fp16). Note that FP16 implementation in cuDNN is currently not optimized for 3D convolutions so results might be worse than FP32.

We recommend running debug version first to make sure all asserts in the code are enabled.

The following scripts demonstrate how to properly read and pre-process images for the Stereo DNN:

Using OpenCV (C++ version is in the `sample_app` as well):
```python
import numpy as np
import cv2

# Using OpenCV
img = cv2.imread('left.png')
img = cv2.resize(img, (1025, 321), interpolation = cv2.INTER_AREA)
# Convert to RGB and then CHW.
img = np.transpose(img[:, :, ::-1], [2, 0, 1]).astype(np.float32)
img /= 255.0
print(img.shape)
with open('left.bin', 'wb') as w:
    img.reshape(-1).tofile(w)

```

Using TensorFlow:
```python
import numpy as np
import tensorflow as tf

img = tf.image.decode_png(tf.read_file('left.png'), dtype=tf.uint8)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize_images(img, [321, 1025], tf.image.ResizeMethod.AREA)
# Convert to CHW.
img_res = np.transpose(img.eval(), [2, 0, 1])
with open('left.bin', 'wb') as w:
    img_res.reshape(-1).tofile(w)

```

Note that due to different implementation of resizing algorithm in TF and OpenCV results will not be byte-wise equal.
