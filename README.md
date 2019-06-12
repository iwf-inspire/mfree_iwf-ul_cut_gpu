# mfree_iwf-ul_cut_gpu

This is the source code to the [publication](https://link.springer.com/article/10.1007/s00170-019-03410-0) **Metal Cutting Simulations using Smoothed Particle
Hydrodynamics on the GPU** to published in the Journal of Advanced Manufacturing Technology ([JAMT](https://www.springer.com/engineering/industrial+management/journal/170)). The purpose of this package is to simulate orthogonal metal cutting operations using Smoothed Particle Hydrodynamics (SPH), parallelized on the GPU using CUDA. This package features some unique characteristics not available in commercial meshfree solvers like the ones available in LSDYNA or ABAQUS, including:

* Stabilization of the solution using the complete array of techniques presented in Gray, J. P., J. J. Monaghan, and R. P. Swift. "SPH elastic dynamics." Computer methods in applied mechanics and engineering 190.49-50 (2001): 6641-6662. 
* Thermal solver using either Particle Strength Exchange or the Brookshaw Approximation, including thermal contact between the tool and the workpiece. 
* Parallelization on the GPU

Parallelization on the GPU allows for dramatic performance benefits compared to conventional solver packages. Some typical timings on the Tesla P100 are given below. The timings are for 1mm of cutting at a cutting speed of 70m/min. 

**Double Precision**

| Resolution      | Adiabatic     | Thermal  | Thermal Contact |
| ---------------:|--------------:| --------:| ---------------:|
|            ~6'000|           457|       459|              807|
|          ~150'000|         6'489|     9'661|           11'372|

**Single Precision**

| Resolution      | Adiabatic     | Thermal  | Thermal Contact |
| ---------------:|--------------:| --------:| ---------------:|
|           ~6'000|            424|       465|              465|
|         ~150'000|          4'793|     6'450|            7'248|

Runtimes are in seconds and valid for the thermal solver using Brookshaw approximation.

Result frames can be viewed using [ParaView](https://www.paraview.org/) using the legacy VTK format. Some typical result frames are shown below:

Preliminary Benchmark, Rubber Rings, Color is Von Mises Stress:

![rings](https://raw.githubusercontent.com/mroethli/mfree_iwf-ul_cut_gpu/master/img/rings.png)

Preliminary Benchmark, Plastic-Plastic Impact, Color is Equivalent Plastic Strain:

![impact](https://raw.githubusercontent.com/mroethli/mfree_iwf-ul_cut_gpu/master/img/impact.png)

Orthogonal Metal Cutting, about 6'000 Particles, Color is Equivalent Plastic Strain::

![lores_cut](https://raw.githubusercontent.com/mroethli/mfree_iwf-ul_cut_gpu/master/img/lores_cut.png)

Orthogonal Metal Cutting, about 500'000 Particles, visualization using [SPLASH](http://users.monash.edu.au/~dprice/splash/), Temperature Field:

![hires_cut](https://raw.githubusercontent.com/mroethli/mfree_iwf-ul_cut_gpu/master/img/hires_cut.png)


**mfree_iwf-ul_cut_gpu** was tested on various versions of Ubuntu Linux. The only dependency is [GLM](https://glm.g-truc.net/0.9.9/index.html). Make files for both a Release version and a Debug build are provided. Tested under NVCC with the GCC suite as host compiler, but any C++11 compliant host compiler should suffice. **mfree_iwf-ul_cut_gpu** was devleloped at [ETHZ](www.ethz.ch) by the following authors

* Matthias Röthlin, mroethli@ethz.ch
* Hagen Klippel, hklippel@ethz.ch
* Mohamadreza Afrasiabi, afrasiabi@ethz.ch

**mfree_iwf-ul_cut_gpu** is free software and licensed under GPLv3
