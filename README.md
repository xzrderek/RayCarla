# RayCarla the Large Scale Autonomous Driving Research Platform.

  **[QuickStart](#QuickStart)**
| **[Development](#Development)**
| **[Results](#Results)**

RayCarla is a **large scale** autonomous driving research platform. 
* Carla simulator.
* Carla on Ray to support distributive simulation (WIP).
* Imitative Learning on Ray to support distributive deep learning. (mutli model training)
* Reinforcement Learning on Ray (WIP).
* Autonomous Driving Interpretability (WIP).
* Leverage Ray Dataset for pipelining... (Planning).

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
   Add following to ~/.bashrc
    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)" 
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```
3. Restart your shell
    ```bash
    exec $SHELL
    ```
4. Install python 3.7 for taining and 3.5 for simulation
    ```bash
    sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```
    ```bash
    pyenv install 3.7.12
    pyenv install 3.5.10
    ```
    *Latest Carla support on 3.7 are work in progress.*
5. Clone code
    ```bash
    git clone https://github.com/xzrderek/RayCarla
    ```
6. Setup virtualenv

    **Simulation venv**

    ```bash
    pyenv local 3.5.10
    pip install virtualenv
    virtualenv ~/venv/3.5
    ```
    ```
    cd RayCarla
    source ~/venv/3.5/bin/activate
    pip install --upgrade pip setuptools
    pip install -r requirements-3.5.txt
    ```
    **Training venv**

    ```bash
    pyenv local 3.7.12
    pip install virtualenv
    virtualenv ~/venv/3.7
    ```
    ```
    cd RayCarla
    source ~/venv/3.7/bin/activate
    pip install --upgrade pip setuptools
    pip install -r requirements-3.7.txt
    ```

7. Install Carla 
    ```bash
    source ~/venv/3.5/bin/activate
    export CARLA_ROOT=~/CARLA
    mkdir -p $CARLA_ROOT
    wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
    tar -xvzf CARLA_0.9.6.tar.gz -C $CARLA_ROOT
    python -m easy_install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg 
    ```
    
8. Install Bazel (optional)
    ```bash
    sudo apt install curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
    sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    sudo apt update && sudo apt install bazel
    ```
9. Training
    ```
    source ~/venv/3.7/bin/activate
    source env.sh
    nohup ./train.sh
    ```

10. Benchmark
    ```
    source ~/venv/3.5/bin/activate
    source env.sh
    nohup python experiment/bench-rip.py
    and other experiments.
    ```
## Results
