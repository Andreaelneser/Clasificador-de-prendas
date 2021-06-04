#!/usr/bin/env python
# coding: utf-8

# In[119]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[120]:


#Cargar los datos
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[121]:


#Tipos y tamaños de datos
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('x_test shape:', x_test.shape)


# In[122]:


#Veamos una  imagen
index = 22
x_train[index]


# In[123]:


img = plt.imshow(x_train[index])


# In[124]:


#Obtengamos la etiqueta
print("La etiqueta de la imagen es:", y_train[index]) 


# In[125]:


clasi = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot'] #creamos esta lista para poder reconocer cada cosa por nombre, no número m
print('El tipo de imagen es:', clasi[y_train[index]])


# In[126]:


#Convertir las etiquetas en un set de 10 números, pues tenemos 10 etiquetas
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# In[127]:


print(y_train_one_hot) #La columna que tenga el valor de 1 será la etiqueta de la imagen


# In[128]:


print(y_train_one_hot[index]) #El set de 10 numeros. Podemos identificar que sí está etiquetándolo bien contando


# In[129]:


#Se normalizan los pixeles para que tengan valores entre 0 y 1
x_train = x_train/255
x_test = x_test/255
x_test= x_test.reshape(-1,28,28,1)


# In[130]:


x_train[index]
x_train= x_train.reshape(-1,28,28,1)


# In[131]:


#Creación del modelo
model = Sequential()

#Primera capa (es de convolucion, extrae caracteristicas de la fotografia de entrada)
model.add(Conv2D(32, (5,5), activation = "relu", input_shape=(28,28,1))) #para analizar el grupo, usamos 32 kernel

#Capa de agrupación
model.add(MaxPooling2D(pool_size = (2,2))) #se obtienen los elementos máximos de los mapas de las caracteristicas, reduciendo la imagen 

model.add(Conv2D(32, (5,5), activation = "relu")) 

model.add(MaxPooling2D(pool_size = (2,2))) 

#Capa de aplanado
model.add(Flatten()) #reduce dimensionalidad

#Capa de 500 neuronas
model.add(Dense(500, activation="relu")) #capas de neuronas tradicionales

#Capa de abandono
model.add(Dropout(0.5)) #se evita el overfitting

#Capa de 250 neuronas
model.add(Dense(250, activation="relu"))

#Capa de 10 neuronas
model.add(Dense(10, activation="softmax"))


# In[132]:


model.compile(loss= 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"]) #buen optimizador para poca memoria y muchos datos


# In[133]:


#Entrenamiento
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=20, validation_split = 0.2)


# In[134]:


#Evaluar el modelo
model.evaluate(x_test, y_test_one_hot)[1]


# In[135]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc = "upper right")
plt.show()


# In[178]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Perdidas')
plt.ylabel('Perdidas')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc = "upper right")
plt.show()


# In[198]:


#Ejemplo
n_img= plt.imread('saco.jpg')
img = plt.imshow(n_img)


# In[199]:


from skimage.transform import resize
resized_img = resize(n_img,(28,28,1))
img = plt.imshow(resized_img)


# In[200]:


predi = model.predict(np.array([resized_img]))
predi


# In[201]:


list_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predi

for i in range (10):
    for j in range(10):
        if x[0][list_ind[i]] > x[0][list_ind[j]]:
            temp = list_ind[i]
            list_ind[i] = list_ind[j]
            list_ind[j] = temp
print(list_ind)


# In[202]:


for i in range(5):
    print(clasi[list_ind[i]], ':', predi[0][list_ind[i]] * 100, '%')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




