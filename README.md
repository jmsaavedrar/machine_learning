# Machine Learning
# Exammples with CNNs
## Dependencies 
- Install Anaconda 
https://docs.anaconda.com/free/anaconda/install/
- Create Environment

  $ conda create -n tf python=3.9

  $ conda activate tf


 
- Install Tensorflow  +  CUDA
  
  Linux
  https://www.tensorflow.org/install/pip#linux_setup
  
  Windows
  https://www.tensorflow.org/install/pip#windows-native 
## Example for MNIST
  Training  a classification CNN
  
  $ python example_convnet.py

  ![image](https://github.com/jmsaavedrar/machine_learning/assets/8441460/214cf77c-fa2d-4c74-8bad-fa07e51dc880)

  At the end, you will have the folder "mnist_model" with the model saved.
  
## Testing features learned by the CNN
Now you can use the model for similarity search in other dataset like [EMNIST-LETTERS](https://www.nist.gov/itl/products-and-services/emnist-dataset).
To test this example, please, download the emnist dataset in npy format from [here](https://www.dropbox.com/scl/fi/kyecjtg2y8w1gmpu1fuai/emnist_data.zip?rlkey=cmhqmp74mz4kmkxupehzz0hw4&dl=0).

After downloading you will have two files:
* data/test_emnist_images.npy
* data/test_emnist_labels.npy

Now, you can try this:

$ python ssearch_emnist.py

After running, you will have some image retrieval results. Please see the images named "result_<id>.png". Examples of these results are:


![result_1415](https://github.com/jmsaavedrar/machine_learning/assets/8441460/ca0033d6-bb11-46c7-bfaf-1b21905b283c)

 ![result_1047](https://github.com/jmsaavedrar/machine_learning/assets/8441460/d9c204b4-71da-4bce-9c29-368f50b5c53e)
 
![result_4843](https://github.com/jmsaavedrar/machine_learning/assets/8441460/7c06baa1-c1b5-422e-90f2-604b697ab9aa)

This shows that we can use a pretrained model for other different problem, where images share some features.

