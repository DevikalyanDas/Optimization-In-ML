# Impact of Learning Rate, Optimizer, Learning Rate Scheduler and Batch Size on OOD generalization in Deep learning

This repository is for the project of Optimization in ML course by Prof. Sebastian Hack at Saaralnd University.

For the code reproducibility:
1. Run ```data_setting_notebook.ipynb ``` which will download data and then arrange the files accordingly as required by the training code. Please take note of the dependencies and install them accordingly. Basically you require the *zipfile, pathlib, shutil* as dependencies for this part.
2. Once the first part is completed, just run the main.py using the command ```python main.py -bt=128 -opt='ADAM' -lrn_rt=1e-2 -ep=50 -schdlr='step_lr' ``` . 
    Things to keep in mind here: 
    bt: batch size. For this experiment we have used 64 and 128
    opt: optimizer. For this experiment we have used ['ADAM', 'SGD', 'RMSPROP', 'ADAMW']. Please select from this list only and pass
    lrn_rt: learning rate 
    ep: epoch
    schdlr: Learning Rate scheduler. Need to select from ['none','step_lr']
  
 
3. To plot the curves: run ```python plotting_curves.py -lrn_rt=1e-4 ```.

**NB**: All the train and validation results obtained during training have been uploaded in the ```./logs``` folder.
