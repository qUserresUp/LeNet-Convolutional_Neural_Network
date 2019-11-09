Instruction of programming assignments for CSCE489: Machine Learning


Environment Building
--------------------
Please install python packages tqdm using:
"pip install tqdm" or
"conda install tqdm"


Installation of Tensorflow
--------------------------
If you are using Anaconda environment, simply using:
"conda install tensorflow"

For other environments, using:
"pip install tensorflow"

You can refer to https://www.tensorflow.org/install/pip for more details.

Note: by default, you are installing the CPU version of tensorflow. DO NOT
install the GPU version unless you are very familiar with tensorflow.

After you finish the installation, run "test_tf_install.py" to test if
the installation is successful. If you receive
'''
hello Tensoflow!
Version 1.13.1
'''
then you are all set. (1.13 is the recommanded version for tensorflow.)


Dataset Descriptions
--------------------
We will use the Cifar-10 Dataset to do the image classification. The 
CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. There are 50000 training images and 10000 
test images.

Download the CIFAR-10 python version from the website:
https://www.cs.toronto.edu/~kriz/cifar.html
then extract the files.

Follow the instructions in "Dataset layout" to load the data and labels.
The training data and labels are save in 5 files. You will need to integrate
the 5 parts. The image data in each file is a 10000x3072 numpy array. Each 
row of the array stores a 32x32 colour image. The first 1024 entries contain 
the red channel values, the next 1024 the green, and the final 1024 the blue.


Assignment Descriptions
-----------------------
There are total three Python files including 'main.py', 'solution.py' and 
'helper.py'. In this assignment, you only need to add your solution in 
'solution.py' file following the given instruction. However, you might need 
to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like 
load data, show images, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper,py' files. Only try 
to add your code to 'solution.py' file and keep function names and parameters 
unchanged.  


APIs you will need
------------------
tf.layers.conv2d
tf.layers.max_pooling2d
tf.layers.flatten
tf.layers.dense
tf.nn.relu

For the honor section, you may also need:
tf.layers.dropout
tf.layers.batch_normalization

Refer to https://www.tensorflow.org/api_docs/python/tf for more details.


Feel free to email Yaochen Xie for any assistance.
Email address: ethanycx@tamu.edu.