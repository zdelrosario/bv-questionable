# bv-questionable
Companion repo for various articles:

* [cutting2018] del Rosario, Fenrich, and Iaccarino, "Cutting the Double Loop: Theory and Algorithms for Reliability-Based Design Optimization with Statistical Uncertainty" (2018, Accepted) NME, [pre-print](https://arxiv.org/abs/1806.00048)

* [beyond[2019] del Rosario, Fenrich, and Iaccarino, "Beyond Basis Values: Fast Precision Margin with FORM" (2019) 20th AIAA Non-Deterministic Approaches Conference

# Dependencies
Core dependencies are Python with Numpy / Scipy and Matplotlib; I prefer to manage these dependencies through [Anaconda Python](https://www.anaconda.com/what-is-anaconda/). We also use a number of convenience functions available through a [python package](https://github.com/zdelrosario/pyutil), which can be easily disabled if desired.

For the cantilever beam function, we use the [Dakota](https://dakota.sandia.gov/) implementation and optimizer.

For the fancy figures under `/genfig`, we use ggplot2, a package in `R`. These are extremely optional.

# Organization
The repo is organized as follows:

`./               # Root directory`  

`./code/`  
`./code/beam/     # Generates results for cantilever beam problem using MIP,  from [cutting2018]`  
`./code/genfig/   # Generates figures from saved data files,                  from [cutting2018]`  
`./code/tension/  # Generates results for uniaxial tension problem, using MIP from [cutting2018]` 
`./code/form/     # Generates results for cantilever beam problem using BIM,  from [beyond2019]`

`./data/          # Output directory for data files`  

`./images/        # Output directory for result images`  

# Instructions
Scripts under `./code/tension/` start with a `Script parameters` section, which can be used to explore different settings. The cases explored in the paper can be accessed by changing the `MYCASE` variable. These scripts are simply run from a terminal, e.g. `python an_tension.py`.

*TODO* Add instructions for `./code/beam/` files.

The `.Rmd` file in `./code/genfig/` is intended to be run from [Rstudio](https://www.rstudio.com/), which aided in prototyping the figure aesthetics.
