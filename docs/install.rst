Installation
============

Before You Begin
----------------

The following requirements should be fulfilled before installing CoCoMagic:

- Python 3.7
- Docker 24.0.7 or newer
- NVIDIA driver 470.256 or newer
- CUDA version 11.4 or newer

CoCoMagic Setup
---------------

Here, we take autonomous driving system testing as an example.
The setup procedures are given below.

#. Setup CoCoMagic

   .. code-block:: shell
      :linenos:

      git clone https://github.com/stkSGarcia/CoCoMagic.git  # Skip this if you already have the source.
      cd CoCoMagic
      python3.7 -m venv .venv
      source .venv/bin/activate

#. Install InterFuser

   .. code-block:: shell
      :linenos:

      cd ..
      git clone https://github.com/opendilab/InterFuser.git
      cd InterFuser
      pip install -r requirements.txt
      cd interfuser
      python setup.py develop

#. Download and setup CARLA 0.9.10.1

   .. code-block:: shell
      :linenos:

      cd ..
      mkdir carla
      cd carla
      wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.1.tar.gz
      tar -xf CARLA_0.9.10.1.tar.gz
      rm CARLA_0.9.10.1.tar.gz
      cd ..
      easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

#. Install CoCoMagic requirements

   .. code-block:: shell
      :linenos:

      cd ../CoCoMagic
      pip install -r requirements.txt

#. Download pretrained model

   The model can be downloaded at `here <http://43.159.60.142/s/p2CN>`_ and needs to be moved to ``conf``.
