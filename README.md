13.05.2022

Mining Massive Data
Programming Assignment 2

Group 12: 
Daniel Rose 12025566
Roxane Jacob 12030167

This project contains the code for the Programming Assignment 2 for the 
University course Mining Massive Data (Summer Term 2022)
at the University of Vienna.

The root folder contains the file main.py. You can run the program simply
by executing

python main.py

from your command line. Make sure that you run the command from the root folder
and that you have all requirements installed. The necessary requirements can be
found within requirements.txt.

Running the script will perform several experiments and will create the 
corresponding plots as output files. Accuracies etc. will be printed in the command line.
Running the script will take approximately 10 minutes depending on the performance of your machine.

The project is structured as follows:

SDM_PA_Task1/
  - data/
  - output/
  - main.py
  - runner_mnist.py
  - runner_svm_models.py
  - runner_toydata.py
  - svm.py
  - utils.py


data: Contains the input data.

output: Contains the generated plots from the experiments.

main.py: Entry point of the project.

runner_mnist.py: Experiments on the mnist dataset.

runner_svm_models.py: Contains helper functions for the experiments.

runner_toydata.py: Experiments in the tiny and large toy dataset.

svm.py: Contains the implementation of the (parallelized) SVM class and the RFF Feature Class.

utils.py: Contains helper functions for data imports and plotting.
