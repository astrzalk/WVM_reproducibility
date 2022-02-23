# Code for Variance Minimization in Wasserstein Space for Invariant Causal Prediction

## Requirements to run code and installation instructions.
* Requires Julia 1.6.1, Jupyter Notebook/Lab (with IJulia installed), and a R installation (tested on
4.0.2) with both the R libraries `pcalg` and `InvariantCausalPrediction` installed.
* To install the R package pcalg run the following R commands:
```
if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager")
BiocManager::install(version = "3.13")
BiocManager::install(c("graph", "RBGL"))
install.packages(pcalg)
```
* In a notebook or in a julia REPL session (to start one run `julia --project=notebooks`), 
run `import Pkg; Pkg.instantiate()` to install all the packages needed to run
the scripts and code in this project. 

## Data Availability & Reproducing Figures
In the `data/` folder we have already generated the data (stored as
the julia serialization format, .jls) necessary 
to produce all the plots from the paper. 
Each notebook in the `notebooks/` folder contains code
to generate the figures from the paper using the data
from `data/`. Concretely, the following notebooks generate the 
following figures:
* `Generate-Figures.ipynb`, generates Figures 2, 3(not c), 4, 7, 8, 9, 10
* `Figure-3c.ipynb`, generates Figure 3c. 
* `Figure-6.ipynb`, generates Figure 6.
* `educational-attainment.ipynb`, generates Figure 11.


## Reproducing Data from Experiments
To reproduce the data found in `data/` 
that is to regenerate the files in the `data/` folder perform the following
while in the base directory of the project:
* `cd src`
* run `julia --project=../notebooks/ GenerateSimulations.jl`
* Moreover, to regenerate the results found in data/results/ 
run `julia --project=../notebooks/ FILE.jl`, where FILE is
one of the following scripts:
    * AlphaVaries.jl -- Generates results for Figure 8.
    * Benchmarking.jl -- Generates results for Figures 2 and 7.
    * CompareWVMandICP.jl -- Generates results for Figures 3(d), 4, 9, and 10.
    * TimeAndPower.jl -- Generates results for Figures 3(a), 3(b); warning by
    default this runs ICP up to 18 pre-selected variables and takes a very,
    very long time. 
