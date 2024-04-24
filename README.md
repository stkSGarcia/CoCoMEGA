# automatic-potato

## Contents of this package

## Installation

1. Setup automatic-potato

   ```shell
   git clone https://github.com/stkSGarcia/automatic-potato.git
   cd automatic-potato
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
   to `automatic-potato/conf`.

## Usage

1. Run CARLA servers

   ```shell
   ```

2. Run automatic-potato

   ```shell
   python -m impl
   ```

3. Local configurations

## Scenario encoding

## License
