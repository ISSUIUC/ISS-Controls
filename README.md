# ISS-CONTROLS
Illinois Space Society's Large Rocketry Guidance, Navigation, and Controls Team simulation tools repository

We test various control design methodologies here with Python before implementing them in [SILSIM](https://github.com/ISSUIUC/ISS_SILSIM) and [flight software](https://github.com/ISSUIUC/MIDAS-Software).

In order to run these programs, we recommend installing **Anaconda** or **Miniconda**. Run the command: `conda env create -f ISS-GNC.yaml` in your command line to create a custom conda environment and install the required packages.

Our current simulation named **PySIM 6DOF** can be run from `6DOF_RK4/simulation/pysim.py` and was used to develop the controller that was used on **Intrepid-III**, which placed **2nd in the 30K COTS** category at **Spaceport America Cup 2023** and Extended Kalman Filter that ran most recently on **Aether II**. This simulation is being updated to include a model for rocket descent. This will be used to develop flight event logic such as stage separation, air-lit motor ignition, parachute deployment and parachute disreefing.

To see the plots from the simulation, you can run `6DOF_RK4/plotter/analysis.py`, which will show multiple plots for analysis. 

Our Monte-Carlo simulation named **Monte-Python** can be run from `6DOF_RK4/simulation/monte_python.py` and will introduce dispersions to the environment such as wind, fluctuations in air pressure, density, initial position and orientation. We will be adding dispersions on air-lit motor ignition time and separation tip-off dynamics.
