# RayCarla the Large Scale Autonomous Driving Research Platform.

  **[QuickStart](#QuickStart)**
| **[Development](#Development)**
| **[Results](#Results)**

RayCarla is a **large scale** autonomous driving research platform. 
* Carla simulator.
* Carla on Ray to support distributive simulation (WIP).
* Imitative Learning on Ray to support distributive deep learning.
* Reinforcement Learning on Ray (WIP).
* Autonomous Driving Interpretability (WIP).

## QuickStart

Follow [raycarla.ipynb](https://github.com/xzrderek/RayCarla/blob/main/raycarla.ipynb) notebook to enjoy!

## Development

1. Install opengl and cuda
    ```bash
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    sudo apt install nvidia-cuda-toolkit
    ```
2. Install pyenv
    ```bash
    curl https://pyenv.run | bash
    ```
3. Add following to ~/.bashrc
    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)" 
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```
4. Restart your shell
    ```bash
    exec $SHELL
    ```
5. Install virtualenv & setuptools
    ```bash
    pip install virtualenv
    pip install --upgrade pip setuptools
    ```
6. Clone code and setup venv
    ```bash
    git clone https://github.com/xzrderek/RayCarla
    cd RayCarla
    virtualenv venv
    source venv/bin/activate
    ```
7. Install Python depdendencies
    ```bash
    pip install -r requirements.txt
    ```
8. Install Carla 
    ```bash
    export CARLA_ROOT=~/carla
    mkdir -p $CARLA_ROOT
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
    tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT
    python -m easy_install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg 
    ```
9. Install Bazel (optional)
    ```bash
    sudo apt install curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
    sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    sudo apt update && sudo apt install bazel
    ```
## Results
