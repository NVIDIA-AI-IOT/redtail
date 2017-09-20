# NVIDIA Redtail project

Autonomous navigation for drones and ground vehicles using deep learning. Refer to [wiki](https://github.com/NVIDIA-Jetson/redtail/wiki) for more information on how to get started.

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
