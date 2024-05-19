# minus1

# TensorFlow GPU Support

## Using the TensorFlow Docker Image

1. **Install Docker Engine**  
   Follow the instructions here: [Install Docker Engine](https://docs.docker.com/engine/install/).

2. **Install the Latest NVIDIA Driver**  
   For Ubuntu 24.04, refer to: [Install NVIDIA Drivers on Ubuntu 24.04](https://linuxconfig.org/how-to-install-nvidia-drivers-on-ubuntu-24-04).  
   Note: The "Software & Updates" tool might not show the newest driver for RTX 4090.  
   - Install via `sudo apt install nvidia-driver-550` from the PPA repository (enable the PPA driver repository).

3. **Install NVIDIA Container Toolkit**  
   Follow the guide here: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

4. **Follow TensorFlow Docker Instructions**  
   Detailed instructions can be found here: [TensorFlow Docker Install](https://www.tensorflow.org/install/docker).

5. **Build Docker Image**  
   Run from the project root directory: `docker build -t minus1-tensorflow .`

6. **Configure PyCharm to Use Docker**
   1. Go to **Settings** -> **Project** -> **Interpreter Settings**.
   2. Add interpreter -> Docker.
   3. Configure Docker interpreter:
      - Server: Docker
      - Dockerfile: Dockerfile
      - Context Folder: .
      - Alternatively: do not use the dockerfile but the image you built previously
   4. Use the system interpreter.

7. **Edit PyCharm Run Configuration for Python Script to Run**
   1. Select the top-right dropdown: current file.
   2. Click the three dots in the current file option -> Run with parameters.
   3. At the bottom of the configuration -> Docker container settings -> click on Browse (folder symbol).
   4. At the bottom run options -> add `--gpus all`.

8. **Important**
   - The Docker image needs `--gpus all` as a run option to access the GPU.
   - To add new pip-Packages: Update requirements.txt and rebuild docker image (should happen automatically with correct pycharm setup).