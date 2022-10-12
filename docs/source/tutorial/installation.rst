Installation
============

.. _installation:

This page details the install instructions for WIPAR.

Requirements
------------

WIPAR supports Windows and Linux distributions.
First, you need to install a Conda-based Python3 distribution like `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_.
Then follow the install instructions for `Snakemake <https://snakemake.readthedocs.io/en/stable/getting_started/installation.html>`_.

Installation
------------

Currently WIPAR is only available on `Github <https://github.com/michaelschaub/calcium-imaging-analysis>`_, so you need to clone the repository to a folder of your choice. This folder will from now on be referenced as ``/WIPAR``.

.. code-block:: console

    $ git pull git@github.com:michaelschaub/calcium-imaging-analysis.git

You should have the following file structure in your ``/WIPAR`` folder.

.. code-block:: none

    .
    ├── config (▶)     | config files for pipeline runs
    ├── resources (▶)  | experimental data and brain atlases
    ├── results (▶)    | processed data and plots
    ├── ci_lib (🛠)     | python package containing all custom functions for the pipeline steps
    ├── workflow (🛠)   | Snakemake logic like rules, envs and entry scripts
    └── SLURM           | batch files to run on computational clusters

    (▶) Running & configuring pipeline (for User)
    (🛠) Extending pipeline functions (for Developers)

Experimental Data
-----------------

To run WIPAR you first need to provide the experimental data you want to process into the ``/WIPAR/resources/experiment/`` folder in the following way

.. code-block:: none

    ├── ...
    ├── resources
    │   ├── meta
    │   └── experiment
    │       ├── subject1
    │       │   ├── experiment_id1
    │       │   ├── experiment_id2
    │       │   └── ...
    │       ├── subject2
    │       │   ├── experiment_id1
    │       │   ├── experiment_id2
    │       │   └── ...
    │       └── ...
    ├── ...


Each datasets is in a separate folder for each subject and each experiment_id (usual the time and date).
A concrete example for the path of a single dataset is ``/WIPAR/resources/experiment/GN06/2021-01-20_10-15-16``.


Now you can test your installation with the :ref:`example from the tutorial <tutorial>`.

.. note::

    If you encounter problems during the setup, please have a look at the :ref:`trouble shooting<trouble_shooting>`