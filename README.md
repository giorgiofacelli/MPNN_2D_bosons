# Graph Neural Network (GNN) for 2D bosonic systems

During the duration of my second semester as a MSc student, I developed a GNN ML architecture to study continuous-variable boson 
particles in a periodic 2D environment. The code is entirely based on JAX Python package and is was mainly trained via open-source package NetKet. 
The model remains quite general and can be applied to arbitrary system size and spatial dimensionality. 
The repository is organized as follows:
- In **MPNN_model.py** is the MPNN ML model,
- In **MPNN_run.py** is a an exmaple scipt of how to run the model with NetKet,
- In **deepset_model.py** is a simpler deep-NN model,
- In **distances.py** are defined some useful functions to compute distances amongst particles,
- In **stats.py** functions for post-training analysis  of the system. For example, the readial correlation function is included,
- In the *latex* folder I included my final report on the project, as well as some slides explaining the main ideas,
- In the *log_files* folder I included the data used to reproduce the results shown in the report.
