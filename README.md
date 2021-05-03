# HistoTNet

Matlab source code for the paper:

	A. Genovese, M. S. Hosseini, V. Piuri, K. N. Plataniotis, and F. Scotti, 
    "Histopathological transfer learning for Acute Lymphoblastic Leukemia detection", 
    in Proc. of the 2021 IEEE Int. Conf. on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA 2021), 
    June 18-20, 2021.
	
Project page:

[https://iebil.di.unimi.it/cnnALL/index.htm](https://iebil.di.unimi.it/cnnALL/index.htm)
    
Outline:
![Outline](https://iebil.di.unimi.it/cnnALL/imgs/outline_civemsa21all.jpg "Outline")

Citation:

	@InProceedings {civemsa21all,
    author = {A. Genovese and M. S. Hosseini and V. Piuri and K. N. Plataniotis and F. Scotti},
    booktitle = {Proc. of the 2021 IEEE Int. Conf. on Computational Intelligence and Virtual Environments for Measurement Systems and Applications (CIVEMSA 2021)},
    title = {Histopathological transfer learning for Acute Lymphoblastic Leukemia detection},
    month = {June},
    day = {18-20},
    year = {2021},}

Main files:

- (1) PyTorch_HistoNet/pytorch_histonet.py: training of the HistoNet;
- (2) PyTorch_HistoTNet/pytorch_histotnet.py: training of the HistoTNet.

Instructions:

1) cd to "(1) PyTorch_HistoNet" and run "pytorch_histonet.py" to train the HistoNet on the ADP database, implemented according to the paper:

    Mahdi S. Hosseini, Lyndon Chan, Gabriel Tse, Michael Tang, Jun Deng, Sajad Norouzi, Corwyn Rowsell, Konstantinos N. Plataniotis, Savvas Damaskinos
    "Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning"
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11747-11756
    
    Required files:
    
    - (1) PyTorch_HistoNet/db_orig/ADP/img_res_1um_bicubic/ <br/>
    ADP database, split in patches, obtained following the instructions at:
    https://www.dsp.utoronto.ca/projects/ADP/
    e.g., (1) PyTorch_HistoNet/db_orig/ADP/img_res_1um_bicubic/001.png_crop_16.png
    
    - (1) PyTorch_HistoNet/db_orig/ADP/ADP_EncodedLabels_Release1_Flat.csv
    file containing the labels of the ADP database, obtained following the instructions at:
    https://www.dsp.utoronto.ca/projects/ADP/
    
2) cd to "(2) PyTorch_HistoTNet" and run "pytorch_histotnet.py" to train the HistoTNet on the ALL-IDB database for Acute Lymphoblastic Leukemia detection.
    
    Required files:
    
    - (2) PyTorch_HistoTNet/db/ALL_IDB2 <br/>
    ALL-IDB database, obtained following the instructions at:
    https://homes.di.unimi.it/scotti/all/
    e.g., (2) PyTorch_HistoTNet/db/ALL_IDB2/Im001_1.tif
