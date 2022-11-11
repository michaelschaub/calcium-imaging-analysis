.. _tutorial:

Tutorial
========

.. note::
    All commands assume you followed the default install procedure for Snakemake within a conda virtual environment. The root of the cloned WIPAR repository is referred to as ``/WIPAR``.

First Pipeline Run
------------------

To get a first look into WIPAR and verify the installation, lets do a simple test run.
Activate conda virtual environment

.. code-block:: console

    $ conda activate snakemake

Run the pipeline with

.. code-block:: console

    $ snakemake -j4 --use-conda decode_all --configfile config/config_GN.yaml

Parameters (like -j4, --use-conda, ...) are detailed [here](snakemake_run). To test the pipeline installation the default config (``/WIPAR/config/config.yaml``) was used together with the config that characterizes Gerions experiment (``/WIPAR/config/config_GN.yaml``). How to customize your config file to meet your requirements is described here.

Inspect results
---------------

To get a quick overview of all the processing steps and produced results from the pipeline run, create a report with

.. code-block:: console

    $ snakemake --report report.html

Open the created ``/WIPAR/report.html`` file in your browser. It should look like this `report <_pages/report.html>`_. The report is explained in :ref:`report<report>`.

Feature Calculation
-------------------

To run the pipeline only upto the feature calculation run

.. code-block:: console

    $ snakemake -j4 --use-conda feature --configfile config/config_GN.yaml


More details regarding the different features and how they are calculated can be found `here<workflow/feature>`. How to import these features for external processing or plotting is explained within an `example jupyter notebook`_.

Neural Decoding
---------------

Decoding_performance performs neural decoding with full feature space and plots results across all features and parcellations

Connectivity Biomarker
----------------------

* `reduce_biomarkers` performs recursive feature elimination to select most discriminative features and visualizes them in an interactive glassbrain plot


Cluster Deployment
------------------
* For usage within a **cluster environment (SLURM)** refer to [this page](cluster)

General guidelines
------------------


.. _example jupyter notebook:

Continue outside of WIPAR
-------------------------------------------

Using WIPAR only for preprocessing and applying external software stacks requires loading the results of a pipeline run externaly. Currently there are two different ways to do so:

1. Using the ci_lib library with the loading methods provided by the corresponding classes
2. Using the export functionality of the pipeline, where commonly used results can be exported as common formats such as csv, npy or npz. 



The Jupyter Notebook ``example_notebook.ipynb`` shows both ways exemplary. Parcellations are loaded with the corresponding ``DecompData``-Class to obtain the labels of the different parcels for plotting. The feature values are loaded from a dictionary that is produced by calling the ``feature`` rule of the pipeline. This rule runs the pipeline upto the feature calculation for all possible combinations of the configuration and aggregates all outputs in a single dictonary (called ``feats_hashxyz.npy``).  

To simplify the usage of this notebook a conda environment with all required packages (including Juypter itself) is provided. To create the env run the following command in the root of the repository.

.. code-block:: console

    $ conda env create --file=examples/env_with_jupyter.yml
    $ conda activate jupyter_WIPAR


.. note:: 
    
    Saving as csv is currently only supported for 2D features (as the default saving method from python doesn't support more dimensions )
    Saving as npz gives weird IO.buffer errors
    Just use npy for now