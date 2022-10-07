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

    $ snakemake -j4 --use-conda decode_all

Parameters (like -j4, --use-conda, ...) are detailed [here](snakemake_run). To test the pipeline installation the default config (``/WIPAR/config/config.yaml``) was used. How to customize your config file to meet your requirements is described here.

To get a quick overview of all the processing steps and produced results from the pipeline run, create a report with

.. code-block:: console

    $ snakemake --report report.html

Open the created ``/WIPAR/report.html`` file in your browser. It should look like this `report <https://raw.github.com/michaelschaub/calcium-imaging-analysis/blob/readthedocs/report.html>`_. The report is explained in :ref:`report<report>`.


Neural Decoding
---------------

        * `decoding_performance` performs neural decoding with full feature space and plots results across all features and parcellations

Connectivity Biomarker
----------------------

* `reduce_biomarkers` performs recursive feature elimination to select most discriminative features and visualizes them in an interactive glassbrain plot


Cluster Deployment
------------------
* For usage within a **cluster environment (SLURM)** refer to [this page](cluster)

General guidelines
------------------


Generating Documentation
------------------------

sphinx-build -b html docs_source/source docs
