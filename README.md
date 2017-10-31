# NVIDIA Redtail project

Autonomous navigation for drones and ground vehicles using deep learning. Refer to [wiki](https://github.com/NVIDIA-Jetson/redtail/wiki) for more information on how to get started.

This project contains deep neural networks, computer vision and control code, hardware instructions and other artifacts that allow users to build a drone or a ground vehicle which can autonomously navigate through highly unstructured environments like forest trails, sidewalks, etc. Our TrailNet DNN for visual navigation is running on NVIDIA's Jetson embedded platform. Our [arXiv paper](https://arxiv.org/abs/1705.02550) describes TrailNet and other runtime modules in detail.

The project's deep neural networks (DNN) can be trained from scratch using publicly available data. A few [pre-trained DNNs](../blob/master/models/pretrained/) are also available as a part of this project. In case you want to train TrailNet DNN from scratch, follow the steps on [this page](./Models).

## References and Demos
* [arXiv paper](https://arxiv.org/abs/1705.02550)
* GTC 2017 talk: [slides](http://on-demand.gputechconf.com/gtc/2017/presentation/s7172-nikolai-smolyanskiy-autonomous-drone-navigation-with-deep-learning.pdf), [video](http://on-demand.gputechconf.com/gtc/2017/video/s7172-smolyanskiy-autonomous-drone-navigation-with-deep-learning%20(1).PNG.mp4)
* [Demo video showing 250 m autonomous flight, DNN activation and control](https://www.youtube.com/watch?v=H7Ym3DMSGms)
* [Demo video showing our record making 1 kilometer autonomous flight](https://www.youtube.com/watch?v=USYlt9t0lZY)
* [Demo video showing generalization to ground vehicle control and other environments](https://www.youtube.com/watch?v=ZKF5N8xUxfw)

# News
* **2017-10-12**: added full simulation Docker image, experimental support for APM Rover and support for MAVROS v0.21+.
  
  * Redtail simulation Docker image contains all the components required to run full Redtail simulation in Docker. Refer to [wiki](../../wiki/Testing-in-Simulator) for more information.
  * Experimental support for APM Rover. Refer to [wiki](../../wiki#platforms) for more information.
  * Several other changes including support for MAVROS v0.21+, updated Jetson install script and few bug fixes.

* **2017-09-07**: NVIDIA Redtail project is released as an open source project.
  
  Redtail's AI modules allow building autonomous drones and mobile robots based on Deep Learning and NVIDIA Jetson TX1 and TX2 embedded systems.
  Source code, pre-trained models as well as detailed build and test instructions are released on GitHub.

* **2017-07-26**: migrated code and scripts to [JetPack 3.1](https://developer.nvidia.com/embedded/jetpack) with [TensorRT 2.1](https://developer.nvidia.com/tensorrt).
  
    TensorRT 2.1 provides significant improvements in DNN inference performance as well as new features and bug fixes. This is a breaking change which requires re-flashing Jetson with JetPack 3.1.
