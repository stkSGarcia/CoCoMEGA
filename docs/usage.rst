Usage
=====

Basic Usage for Test Case Generation
------------------------------------

Here, we take autonomous driving system testing as an example.

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

      python -m impl search -a ['ccea' | 'ga' | 'rs']

#. Use ``--help`` for more options

   .. code-block:: shell
      :linenos:

      python -m impl --help

Incorporate Runtime Scenarios
-----------------------------

If you apply constraints or runtime initialization to the search, you must also generate a set of representative runtime scenarios.

#. Collect runtime data

   To collect runtime data, use the command below:

   .. code-block:: shell
      :linenos:

      python -m impl collect_runtime_data

#. Convert runtime data to runtime scenarios

   .. code-block:: shell
      :linenos:

      python -m impl convert -d [dir_of_runtime_data]

Train You Own InterFuser
------------------------

If you are performing differential testing, you need to train an additional InterFuser model.

#. Generate training data

   To generate training data using a rule-based agent, use the command below:

   .. code-block:: shell
      :linenos:

      python -m impl generate_train_data

#. Train InterFuser

   To train an Interfuser model, use the command below:

   .. code-block:: shell
      :linenos:

      python -m impl train

Build Your Own Domain
---------------------

Please refer to :doc:`guide`.
