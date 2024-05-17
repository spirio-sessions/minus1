# minus1

# TensorFlow GPU Support

## Using the TensorFlow Docker Image

1. **Install Docker Engine**  
   Follow the instructions here: [Install Docker Engine](https://docs.docker.com/engine/install/)

2. **Install the Latest NVIDIA Driver**  
   For Ubuntu 24, refer to: [Install NVIDIA Drivers on Ubuntu 24.04](https://linuxconfig.org/how-to-install-nvidia-drivers-on-ubuntu-24-04)  
   Note: The "Software & Updates" tool did not show the newest driver for RTX 4090.  
   -> Install via `sudo apt install nvidia-driver-550` from the PPA repository (enable the PPA driver repository).

3. **Install NVIDIA Container Toolkit**  
   Follow the guide here: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

4. **Follow TensorFlow Docker Instructions**  
   Detailed instructions can be found here: [TensorFlow Docker Install](https://www.tensorflow.org/install/docker)

5. **Build docker image**    
   Run from project root dir: `docker build -t minus1-tensorflow .`

6. **Configure PyCharm to use Docker**
   1. settings -> project -> interpreter settings
   2. Add interpreter -> Docker
   3. Configure Docker interpreter
      - Server: Docker
      - Dockerfile: Dockerfile
      - Context Folder: .
   4. Use system interpreter

7. **Edit Pycharm rum configuration for python script to run**
   1. Top right dropdown: current file
   2. 3 dots in current file option -> Run with parameters
   3. at bottom of configuration -> Docker container settings -> klick on Browse (folder symbol)
   4. at bottom run options -> add `--gpus all`

7. **Important**
   - The docker image needs `--gpus all` as run option to see gpu