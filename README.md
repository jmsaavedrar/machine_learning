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
  
