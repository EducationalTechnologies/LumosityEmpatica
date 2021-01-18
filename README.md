# LumosityEmpatica
Script for analysing the Lumosity sensor data collected with Empatica E4. 
It uses the MLT-JSON sensor format and it's based on https://github.com/dimstudio/SharpFlow 

## Setup the Python enviroment
1. Git clone the repository on the local enviroment
2. Install Anaconda https://www.anaconda.com/products/individual 
3. Install PyCharm https://www.jetbrains.com/pycharm/download/
4. Open this project in PyCharm 
5. Add Conda as interpreter 
6. Resolve dependencies in case some libraries are not installed

## How to start
1. Create a new folder in ther root called `manual_sessions`
2. Add a dataset (zip sessions) in new folder `manual_sessions/lumosity-empatica`
3. Run `dataset_descriptor.py`
