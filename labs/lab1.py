#tensores

import numpy as np
import skimage.io as io
import matplotlib.pyplot  as plt
#Creación de tensores (arrays)

#1. Crear un array 1D (vector)

a = np.array([2,4,6, 8])
print(a.shape)
print(a.dtype)
b = a + 2 
print(b)
ones = np.ones(a.shape)
ones = np.ones(a.shape, dtype = a.dtype)
print(ones.dtype)
c = a + ones 
print(c)

# 2. crear tensor 2D (matriz)
"""
1 2 
3 4
5 6 
7 8
""" 
matrix = np.array([[1,2], 
                   [3,4], 
                   [5,6], 
                   [7,8]])
print(matrix) 
print(matrix.shape)
"""
Podriamos leer una imagen y mostrar el contenido
sería una imagen en escala de grises de modo que tenga un solo canal
podemos oscurecerla o aclararla
"""
filename ='rice.jpg' 
image = io.imread(filename)
print(image.dtype)
image = image.astype(np.int32)
image = np.minimum(np.maximum(image + 50, 0), 255)
#image [image < 0 ] = 0
#image [image > 255 ] = 255
print(image.shape)
plt.imshow(image, cmap = 'gray', vmax = 255, vmin = 0)
plt.show()
# 3. crear tensor 3D (matriz)
"""
Podríamos leer una imagen a colores RGB (HxWxC)
"""
filename ='flower.jpg' 
image = io.imread(filename)
print(image.shape)
plt.imshow(image)
plt.show()
# 4. Operar con tensores 
# revisar * , matmul, dot 

# Ejercicio (calcular distancia euclideana entre vectores)
"""
La idea es trabajar en R^10, con datos aleatorios
D = array de Nx10
q = array de 1x10
Calcular la distancia euclideana entre todos los D x q (usando solo algebra matricial)
""" 

#Pensemos en un ejercicio para ellos

#Plot (scatter)