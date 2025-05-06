Usage
=====

Basic Usage
-----------

#. Build docker image

   .. code-block:: shell
      :linenos:

      docker build -t [tag_name] .

   The ``tag_name`` should be consistent with that defined in the configuration file located in ``docker:image``
   (default is ``carla-0.9.10``).

#. Local configurations

   Customize your configurations by modifying the default ``config.yaml`` file located in the ``conf`` directory.
   Save the customized configuration file as ``config.yaml`` in your project's root directory.

#. Run algorithms

   .. code-block:: shell
      :linenos:

      python -m impl search -a [algorithm]

#. Use ``--help`` for more options

   .. code-block:: shell
      :linenos:

      python -m impl --help

Collect Runtime Data
--------------------

To collect runtime data, use the command below:

.. code-block:: shell
   :linenos:

   python -m impl collect_runtime_data

Generate Training Data
----------------------

To generate training data using a rule-based agent, use the command below:

.. code-block:: shell
   :linenos:

   python -m impl generate_train_data

Train Interfuser
----------------

To train an Interfuser model, use the command below:

.. code-block:: shell
   :linenos:

   python -m impl train
