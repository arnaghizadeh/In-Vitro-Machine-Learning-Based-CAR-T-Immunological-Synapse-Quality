### In Vitro Machine Learning-Based CAR T Immunological
Official implementation of the In Vitro Machine Learning Based CAR T Immunological Synapse Quality Measurements Correlate with Patient Clinical Outcomes. The code is written with Python. With this code, you will get an overall structure of the algorithm for training and testing.

 ### Parameters
There are many arguments that can be passed to main.py and test.py. For convenience, the most important ones are in P.py. The pretrained_weight can be downloaded from https://drive.google.com/file/d/14S5FM_ToWBW205fZ8gx2NzB5irxilU4r/view?usp=sharing and postrained_weight can be downloaded from https://drive.google.com/file/d/1ANgtRILhAahkErYWsldRjx1xChyaNZig/view?usp=sharing. After download, the folder's path -and not file itself- should be set in P.py on the pretrained_weights, postrained_weights variables. 



### Antigen Dataset
The Antigen dataset used to test and train the model can be aqcuired from here: https://drive.google.com/file/d/1EQUGgNJovywaqJKFFU-Z4zFPs0bx3_n0/view?usp=sharing



### Citation

If you used this code and the dataset in your academic work, please consider citing the accompanying paper. For non-academic use, please contact professor Dongfang Liu [dongfang.liu at rutgers dot edu]:




### References
For the code we did get help from these two repositories: [AutoAugment](https://github.com/tensorflow/models/tree/master/research/autoaugment), [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar).
