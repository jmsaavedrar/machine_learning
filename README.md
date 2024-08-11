# Machine Learning
# Examples with CNNs
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
Now, you can use the model for similarity search in other dataset like [EMNIST-LETTERS](https://www.nist.gov/itl/products-and-services/emnist-dataset).
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

This shows that we can use a pretrained model for other different but similar problem. We say similar because images comes from the same nature. In this case, letters from emnist and digits from mnist are all handwritten symbols.

In addition, you can visualize the feature space projecting the original features to 2D points using UMAP (see umap_view.py). Below is an example of the space representation on emnist dataset.

![emnist_plot](https://github.com/jmsaavedrar/machine_learning/assets/8441460/121531bc-8b65-4e46-aeef-f55b60bde823)



## Required Data
-[mnist_model](https://www.dropbox.com/scl/fi/ovucxclytv4m72xnvgsb0/mnist_model.zip?rlkey=jzzaw6azetqmt2xrh0qigvxtu&st=hleis61y&dl=0)
-[emnist_data](https://www.dropbox.com/scl/fo/y4jw1onrveqvamunqz7ev/AP3rDCddbreej7g_6hXNG-8?rlkey=r0qal2pt9qowai2j2lad2bvdu&st=xjzt26wk&dl=0)
