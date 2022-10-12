File Structure
==============

| (▶) Running & configuring pipeline (for User)
| (🛠) Extending pipeline functions (for Developers)

.. code-block:: none

    .
    ├── ci_lib (🛠)    | custom python package containing all custom functions for the pipeline steps
    ├── workflow (🛠)  | Snakemake logic like rules, envs and entry scripts
    ├── config (▶)    | config files for pipeline runs
    ├── resources (▶) | experimental data and brain atlases
    ├── results (▶)   | processed data and plots
    └── SLURM     | batch files to run on computational clusters