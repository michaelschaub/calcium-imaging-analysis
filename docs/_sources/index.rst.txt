WIPAR
=====

**WIPAR** (**W**\ idefield **I**\ maging **P**\ ipeline for **A**\ nalysis and **R**\ egression) is a data pipeline for processing and analysing task-specific (widefield) calcium imaging data through neural decoding.
Here, calcium activity is a proxy for neuronal activations.
It provides stand-alone functionalities to visualize the data analysis as well as enabling the export of processed data for other visualization purposes.


----------------
Feature Overview
----------------

* :doc:`Brain Alignment of different subjects and experiments<workflow/brain_alignment>`
* :doc:`Different data-driven & anatomical parcellations (Atlas-based, locaNMF, SVD, ICA)<workflow/parcellation>`
* :doc:`Different brain connectivity measurements (FC, EC)<workflow/brain_connectivity>`
* :doc:`Neural Decoding through simple machine learning models<workflow/neural_decoding>`
* :doc:`Biomarker Detection through Feature Selection<workflow/biomarker_detection>`
* :doc:`Generic & customizable selection of trials and conditions<workflow/trials_conditions>`
* :doc:`Interactive & automated visualization of results<workflow/visualizing>`
* :doc:`Web-interface to review pipeline runs and export results<workflow/report>`


.. _main-getting-started:

---------------
Getting started
---------------

* To get a first impression of the results you can generate with WIPAR, please have a look at an exemplary `report <https://raw.github.com/michaelschaub/calcium-imaging-analysis/blob/readthedocs/report.html>`_ that is automatically compiled.
* To get started :ref:`install WIPAR and it's requirements <installation>`
* For an example on how to use WIPAR and general guidelines have a look at the :ref:`tutorial <tutorial>`
* Detailed explanations of the configurations can be found here




.. toctree::
   :caption: Getting started
   :name: getting_started
   :maxdepth: 1
   :hidden:

   tutorial/installation
   tutorial/tutorial

.. toctree::
   :caption: Workflow
   :name: workflow
   :maxdepth: 1
   :hidden:

   workflow/loading
   workflow/brain_alignment
   workflow/parcellation
   workflow/brain_connectivity
   workflow/trials_conditions
   workflow/neural_decoding
   workflow/biomarker_detection
   workflow/visualizing
   workflow/report


.. toctree::
   :caption: Development
   :name: develop
   :hidden:

   development/file_struct

.. toctree::
   :caption: API
   :name: api
   :hidden:

    autoapi/index