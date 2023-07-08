# mnist-digit-classifier
 A MNIST classifier and an accompanying GUI, built with Python 3.8.16, Pytorch, numpy

 Feel free to take the model and gui and run with it, just GPU GPL things etc. etc.

 The plan is gonna be to retrain this model with some failing testcases to improve performance

 This model is striaght of of MNIST training right now (~96% accuracy on some test set)

 Details:
 Miniconda
 Python 3.8.16

 How to run:
 clone repository and cd into it
 run `conda env create -f environment.yml` in commandline where you want your conda env created
 run `conda activate name_of_environment` (not sure what this will be didn't test it)
 run the python script mnist_gui.py (I think `python mnist_gui.py` should work)
 happy drawing!

 Known Issues:

 Model legit does not know what a six looks like

 Plans:
 Auto input image correction (scaling, rotation, norming) [probably another model to train for this purpose kekw]
 Further fine tuning using SOTA techniques (validation, boosting, etc.) [I gotta learn this too lol]