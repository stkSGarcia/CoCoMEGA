# CoCoMEGA

This is the replication package for the paper titled *Using Cooperative Co-evolutionary Search to Generate Metamorphic
Test Cases for Autonomous Driving Systems* by Hossein Yousefizadeh, Shenghui Gu, Lionel C. Briand, and Ali Nasr.

## Contents of this package

1. Implementation of CoCoMEGA, Standard Genetic Algorithm (SGA), and Random Search (RS).
2. Installation and usage guidelines.
3. Metamorphic relations involved.
4. The postprocess script used to answer RQ1 and RQ2, provided in `test/postprocess.ipynb`.
5. A copy of the evaluation results provided in `results.tar.gz`.

## Installation

### Prerequisites

- Python 3.7.
- Docker 24.0.7 or newer.
- NVIDIA driver 470.256 or newer.
- CUDA version 11.4 or newer.

### Steps

1. Setup CoCoMEGA

   ```shell
   git clone https://github.com/stkSGarcia/CoCoMEGA.git
   cd CoCoMEGA 
   python3.7 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Install InterFuser

   ```shell
   cd ..
   git clone https://github.com/opendilab/InterFuser.git
   cd InterFuser
   pip install -r requirements.txt
   cd interfuser
   python setup.py develop
   ```
3. Download and setup CARLA 0.9.10.1

   ```shell
   cd ..
   ./setup_carla.sh
   easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
   ```

4. Download pretrained model

   The model can be downloaded at [here](http://43.159.60.142/s/p2CN) and needs to be moved
   to `conf`.

## Usage

1. Build docker image

   ```shell
   docker build -t [tag_name] .
   ```
   The `tag_name` should be consistent with that defined in the configuration file located in `simulation:docker:image`.

2. Local configurations

   Customize your configurations by modifying the default `config.yaml` file located in the `conf` directory.
   Save the customized configuration file as `config.yaml` in your project's root directory.

3. Run algorithms (CoCoMEGA, SGA, RS)

   ```shell
   python -m impl search -a [algorithm]
   ```

4. Use `--help` for more options

   ```shell
   python -m impl --help
   ```

## Metamorphic Relations

- MR1: If a pedestrian appears on the roadside, then the ego-vehicle should slow down.
- MR2: If the driving time changes into night, then the ego-vehicle should slow down.
- MR3: Adding a vehicle in the front of the ego vehicle, the speed of the ego vehicle should decrease in t1% to t2%.
- MR4: Adding a pedestrian in the front of the ego vehicle, the speed of the ego vehicle should decrease in t1% to t2%.
- MR5: Changing from sunny to rainy, the speed of the ego vehicle should decrease in t1% to t2%.
- MR6: The density of fog (light/heavy/dense/strong/extra strong fog) in a foggy driving scene will not affect the
  steering angle of the autonomous driving systems.
- MR7: No matter how the driving scenes are synthesized to cope with different weather conditions (sunny and rainy), the
  driving steering angle are expected to be consistent with those under the corresponding original driving scenes.

## Collect Runtime Data
To collect runtime data, use the command below
```shell
   python -m impl collect_runtime_data
```

## Generate Training Data
To generate training data using a rule-based agent, use the command below
```shell
   python -m impl generate_train_data
```

## Train Interfuser
To train an Interfuser model, use the command below
```shell
   python -m impl train
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
