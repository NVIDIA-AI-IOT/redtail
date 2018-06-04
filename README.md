# NVIDIA Redtail project

Autonomous visual navigation components for drones and ground vehicles using deep learning. Refer to [wiki](https://github.com/NVIDIA-Jetson/redtail/wiki) for more information on how to get started.

This project contains deep neural networks, computer vision and control code, hardware instructions and other artifacts that allow users to build a drone or a ground vehicle which can autonomously navigate through highly unstructured environments like forest trails, sidewalks, etc. Our TrailNet DNN for visual navigation is running on NVIDIA's Jetson embedded platform. Our [arXiv paper](https://arxiv.org/abs/1705.02550) describes TrailNet and other runtime modules in detail.

The project's deep neural networks (DNNs) can be trained from scratch using publicly available data. A few [pre-trained DNNs](../master/models/pretrained/) are also available as a part of this project. In case you want to train TrailNet DNN from scratch, follow the steps on [this page](../../wiki/Models).

The project also contains [Stereo DNN](../master/stereoDNN/) models and runtime which allow to estimate depth from stereo camera on NVIDIA platforms.

**CVPR 2018**: we will present our work at [CVPR 2018](http://cvpr2018.thecvf.com/) conference as a part of [Workshop on Autonomous Driving](http://www.wad.ai/index.html). Feel free to stop by and chat.

## References and Demos
* [Stereo DNN, CVPR18 paper](https://arxiv.org/abs/1803.09719), [Stereo DNN video demo](https://youtu.be/0FPQdVOYoAU)
* [TrailNet Forest Drone Navigation, IROS17 paper](https://arxiv.org/abs/1705.02550), [TrailNet DNN video demo](https://youtu.be/H7Ym3DMSGms)
* GTC 2017 talk: [slides](http://on-demand.gputechconf.com/gtc/2017/presentation/s7172-nikolai-smolyanskiy-autonomous-drone-navigation-with-deep-learning.pdf), [video](http://on-demand.gputechconf.com/gtc/2017/video/s7172-smolyanskiy-autonomous-drone-navigation-with-deep-learning%20(1).PNG.mp4)
* [Demo video showing 250m autonomous flight with TrailNet DNN flying the drone](https://youtu.be/H7Ym3DMSGms)
* [Demo video showing our 1 kilometer autonomous drone flight with TrailNet DNN](https://youtu.be/USYlt9t0lZY)
* [Demo video showing TrailNet DNN driving a robotic rover around the office](https://youtu.be/lOmT4yWcJrM)
* [Demo video showing TrailNet generalization to ground vehicles and other environments](https://youtu.be/ZKF5N8xUxfw)

# News
* **2018-06-04**: CVPR 2018 workshop. Fast version of Stereo DNN.
  * Presenting our work at [CVPR 2018](http://cvpr2018.thecvf.com/) conference as a part of [Workshop on Autonomous Driving](http://www.wad.ai/index.html).
  * Added fast version of Stereo DNN model based on ResNet18 2D model. The model runs at 10fps on Jetson TX2. See [README](../master/stereoDNN/) for details and check out updated [sample_app](../master/stereoDNN/sample_app).

* **GTC 2018**: Here is our [Stereo DNN session page at GTC18](https://2018gputechconf.smarteventscloud.com/connect/sessionDetail.ww?SESSION_ID=152050) and [the recorded video presentation](http://on-demand.gputechconf.com/gtc/2018/video/S8660/)

* **2018-03-22**: redtail 2.0.
  * Added Stereo DNN models and inference library (TensorFlow/TensorRT). For more details, see the [README](../master/stereoDNN/).
  * Migrated to JetPack 3.2. This change brings latest components such as CUDA 9.0, cuDNN 7.0, TensorRT 3.0, OpenCV 3.3 and others to Jetson platform. Note that this is a breaking change.
  * Added support for INT8 inference. This enables fast inference on devices that have hardware implementation of INT8 instructions. More details are on [our wiki](../../wiki/ROS-Nodes#int8-inference).

* **2018-02-15**: added support for the TBS Discovery platform.
  * Step by step instructions on how to assemble the [TBS Discovery drone](../../wiki/Skypad-TBS-Discovery-Setup).
  * Instructions on how to attach and use a [ZED stereo camera](https://www.stereolabs.com/zed/).
  * Detailed instructions on how to calibrate, test and fly the drone.

* **2017-10-12**: added full simulation Docker image, experimental support for APM Rover and support for MAVROS v0.21+.
  
  * Redtail simulation Docker image contains all the components required to run full Redtail simulation in Docker. Refer to [wiki](../../wiki/Testing-in-Simulator) for more information.
  * Experimental support for APM Rover. Refer to [wiki](../../wiki#platforms) for more information.
  * Several other changes including support for MAVROS v0.21+, updated Jetson install script and few bug fixes.

* **2017-09-07**: NVIDIA Redtail project is released as an open source project.
  
  Redtail's AI modules allow building autonomous drones and mobile robots based on Deep Learning and NVIDIA Jetson TX1 and TX2 embedded systems.
  Source code, pre-trained models as well as detailed build and test instructions are released on GitHub.

* **2017-07-26**: migrated code and scripts to [JetPack 3.1](https://developer.nvidia.com/embedded/jetpack) with [TensorRT 2.1](https://developer.nvidia.com/tensorrt).
  
    TensorRT 2.1 provides significant improvements in DNN inference performance as well as new features and bug fixes. This is a breaking change which requires re-flashing Jetson with JetPack 3.1.
