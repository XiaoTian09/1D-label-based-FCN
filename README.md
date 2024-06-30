# 1D-label-based-FCN

Introduction: We design a Fully Convolutional Network (FCN) to determine the locations for surface microseismic events.

'fcn_train.py' : Location determination network based on FCN. The input is the waveform data with the size of 1001x758x1. The output size is 256x3.
'data.npy' is the training data of 2000 samples with the size of 2000x1001x128x1.
'label.npy' is corresponding to the training label of 2000 samples with the size of 2000x256x3.
The traning data demo (2000 samples,1.4GB) are accessible on the zenodo via https://zenodo.org/records/12597983
Any questions, please contact: tianx@ecut.edu.cn
