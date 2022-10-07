WIPAR
=====

**WIPAR** (**W** idefield **I** maging **P** ipeline for **A** nalysis and **R** egression) is a data pipeline for processing and analysing task-specific (widefield) calcium imaging data through neural decoding.
Here, calcium activity is a proxy for neuronal activations.
It provides stand-alone functionalities to visualize the data analysis as well as enabling the export of processed data for other visualization purposes.


.. _main-getting-started:

---------------
Getting started
---------------

* To get a first impression of the results you can generate with WIPAR, have a look at an exemplary `report <https://raw.github.com/michaelschaub/calcium-imaging-analysis/blob/readthedocs/report.html>`_ that is automatically compiled.
* To get started :ref:`install WIPAR and it's requirements <installation>`
* For an example on how to use WIPAR have a look at the :ref:`tutorial <tutorial>`
* Detailed explanations of the configurations can be found here


.. toctree::
   :caption: Getting started
   :name: getting_started
   :hidden:
   :maxdepth: 1

   tutorial/usage

.. toctree::
   :caption: Workflow
   :name: workflow
   :maxdepth: 1

   workflow/loading
   workflow/parcellation

.. toctree::
   :caption: API
   :name: api
   :hidden:

    autoapi/index