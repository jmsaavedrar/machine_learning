import tensorflow as  tf
import skimage.io as io
import skimage.color as color
import os
import numpy as np 
import matplotlib.pyplot as plt

model_dir = '/home/jsaavedr/Models/model/' 
model = tf.keras.models.load_model(model_dir)


data_dir = '/home/jsaavedr/datasets/berlin/Sketch_EITZ/'
#png_w256/rooster/13885.png    173
#filename = os.path.join(data_dir, 'png_w256/rooster/13885.png') # 186
filename = os.path.join(data_dir, 'png_w256/umbrella/19079.png') # 119
image = io.imread(filename)
plt.imshow(image, cmap = 'gray')
plt.show()
image = color.gray2rgb(image)
image = image.astype(np.float32)
print(np.max(image))
#image = 255 - image/255
image =np.expand_dims(image, axis = 0)
print(image.shape)
ans = model.predict(image)
ans = np.argmax(ans)
print(ans)
#print(ans)


