.. _tutorial:

Tutorial
========

.. note::
    All commands assume you followed the default install procedure for Snakemake within a conda virtual environment. The root of the cloned WIPAR repository is referred to as ``/WIPAR``.

Running the Pipeline
--------------------

To get a first look into WIPAR and verify the installation, lets do a simple test run.
Activate conda virtual environment

.. code-block:: console

    $ conda activate snakemake

Run the pipeline with

.. code-block:: console

    $ snakemake -j8 --use-conda test --configfile config/config_RNG.yaml

Parameters (like -j8, --use-conda, ...) are detailed :doc:`here<tutorial/snakemake>`. By running this command the pipeline is run with the default config (``/WIPAR/config/config.yaml``) extended by the config file (``/WIPAR/config/config_RNG.yaml``). This config file defines random experimental data to be used as quick test of the pipeline installation.

It follows the same structure as the configuration files for Gerions experiment (``/WIPAR/config/config_GN.yaml``) and Simons Experiment (``/WIPAR/config/config_mSM.yaml``). How to customize your config file to meet your requirements is described :doc:`here<tutorial/configuration>`.

Inspect results
---------------

To get a quick overview of all the processing steps and produced results from a pipeline run, create a report with

.. code-block:: console

    $ snakemake test --report report.html

Open the created ``/WIPAR/report.html`` file in your browser. It should look like this `report <_pages/report.html>`_. The report is explained in :ref:`report<report>`. 

If you have a large pipeline run the resulting report is quite large, serving the report with a http server instead is prefered. This can be done by

.. code-block:: console

    $ snakemake test --report report.zip --configfile config/config_RNG.yaml
    $ unzip report.zip; mv report/report.html report/index.html
    $ python -m http.server --directory report


Feature Calculation
-------------------

To run the pipeline only upto the feature calculation run

.. code-block:: console

    $ snakemake -j4 --use-conda feature --configfile config/config_GN.yaml


More details regarding the different features and how they are calculated can be found `here<workflow/feature>`. How to import these features for external processing or plotting is explained within an `example jupyter notebook`_.

Neural Decoding
---------------

Perform neural decoding with full feature space and plots results across all features and parcellations

Run the pipeline with

.. code-block:: console

    $ snakemake -j4 --use-conda decode --configfile config/config_GN.yaml


More info on neural decoding with WIPAR can be found here :doc:`here<workflow/neural_decoding>`


Activity & Connectivity Biomarker
---------------------------------

Perform recursive feature elimination to select most discriminative features and visualizes them in an interactive glassbrain plot

Run the pipeline with

.. code-block:: console

    $ snakemake -j4 --use-conda biomarkers --configfile config/config_GN.yaml

More info on finding biomarkers with WIPAR can be found here :doc:`here<workflow/biomarker_detection>`


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