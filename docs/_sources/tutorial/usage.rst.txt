Usage
=====

To use WIPAR, first install it.

.. _installation:

Installation
------------

First, you need to install a Conda-based Python3 distribution like `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_.
Then follow the install instructions for `Snakemake <https://snakemake.readthedocs.io/en/stable/getting_started/installation.html>`_.

Currently WIPAR is only available on `Github <https://github.com/michaelschaub/calcium-imaging-analysis>`_, so you need to pull the repository to a folder of your choice. This folder will from now on be referenced as root.

.. code-block:: console

    $ cd path/to/root
    $ git pull git@github.com:michaelschaub/calcium-imaging-analysis.git

Experimental Data
-----------------

To run WIPAR you first need to provide the experimental data you want to process.

.. code-block:: console

    root/resources/experiment/"subject"/"experiment_id"/
    e.g. root/resources/experiment/GN06/2021-01-20_10-15-16


Have a look at the :ref:`Trouble Shooting<trouble_shooting>` if you encounter problems during the setup