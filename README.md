# Impact of Learning Rate, Optimizer, Learning Rate Scheduler and Batch Size on OOD generalization in Deep learning

This repository is for the project of Optimization in ML course by Prof. Sebastian Hack at Saaralnd University.

For the code reproducibility:
1. Run ```data_setting_notebook.ipynb ``` which will download data and then arrange the files accordingly as required by the training code. Please take note of the dependencies and install them accordingly. Basically you require the *zipfile, pathlib, shutil* as dependencies for this part.
2. Once the first part is completed, just run the main.py using the command ```pthon main.py -bt=128 -opt='ADAM' -lrn_rt=1e-2 -ep=50 -pt='./logs/new_scheduler' ``` . Things to keep in mind here: bt: batch size, opt: optimizer, lrn_rt: learning rate, ep: epoch and pt: path where all the log files such as csvs and model weights will be stored
