#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.datasets import fashion_mnist


# In[55]:


(train_X,train_Y), (test_X, test_Y) = fashion_mnist.load_data()


# In[ ]:


# Next we analyze how images in the dataset look like

# Even though you know the dimension of the images by now, it's still worth the effort to analyze it programmatically: 
# you might have to rescale the image pixels and resize the images.


# In[56]:


import numpy as np
from keras.utils import to_categorical #convert to binay?
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf 


# In[6]:


print ('Training data shape:', train_X.shape, train_Y.shape)
print ('Testing data shape:', test_X.shape, test_Y.shape)


# In[ ]:


# From the above output, we can see that 
# the training data has a shape of 60000 x 28 x 28 
# since there are 60,000 training samples each of 28 x 28 dimension. 
# Similarly, the test data has a shape of 10000 x 28 x 28 since there are 10,000 testing samples.


# In[7]:


# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes) #len ?
print('Total number of outputs:', nClasses)
print('Output classes:', classes)


# In[ ]:


# (By doing so we can see the features of the dataset. (It stands for 10 classes in total.))

# From the result above, There's a total of ten output classes that range from 0 to 9.

# Also, we can take a look at what the images in your dataset:


# In[57]:


plt.figure(figsize = [5,5])

# Display the first image in training data
plt.subplot(121) #what does it stand for?
plt.imshow(train_X[0,:,:], cmap = 'gray')
plt.title('Ground Truth:{}'.format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:],cmap = 'gray')
plt.title('Ground Truth:{}'.format(test_Y[0]))


# In[ ]:


# The output of above two plots looks like an ankle boot, 
# and this class is assigned a class label of 9. 

# Similarly, other fashion products will have different labels, 
# but similar products will have same labels.

# This means that all the 7,000 ankle boot images will have a class label of 9.


# In[ ]:


# Data Processing

# As we can see in the above plot, 
# the images are grayscale images have pixel values that range from 0 to 255. # why 255? 25*25?

# Also, these images have a dimension of 28 x 28. 
# As a result, we'll need to preprocess the data before you feed it into the model. #why? the reason follows.


# In[ ]:


# As a first step, convert each 28 x 28 image of the train and test set 
# into a matrix of size 28 x 28 x 1 which is fed into the network.


# In[58]:


train_X = train_X.reshape(-1,28,28,1) # why -1?
test_X = test_X.reshape(-1,28,28,1)
train_X.shape, test_X.shape # why 60000,100000


# In[ ]:


# The data right now is in an int8 format, # what is int8?
# so before you feed it into the network you need to convert its type to float32, #why?
# and you also have to rescale the pixel values in range 0 - 1 inclusive. So let's do that!


# In[59]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255. 
test_X = test_X / 255.


# In[ ]:


# Now you need to convert the class labels into a one-hot encoding vector.
# In one-hot encoding, you convert the categorical data into a vector of numbers. 
# The reason why you convert the categorical data in one hot encoding is that machine learning algorithms cannot work with categorical data directly. 
# You generate one boolean column for each category or class. Only one of these columns could take on the value 1 for each sample. 
# Hence, the term one-hot encoding.


# In[60]:


# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:',train_Y[0]) 
print('After conversion to one-hot:',train_Y_one_hot[0])


# In[61]:


# train the model on 80\% of the training data and validate it on 20\% of the remaining training data.

from sklearn.model_selection import train_test_split
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot,test_size = 0.2, random_state= 13)


# In[17]:


# check the shape of training and validation set
train_X.shape, valid_X.shape, train_label.shape, valid_label.shape


# In[ ]:


# The Network (The Architecture of the Model)

# we use 3 convolutional layers:
# 1stlayer: 32 - 3x3 filters
# 2ndlayer: 64 - 3x3 filters
# flatten
# 3rdlayer: 128 - 3x3 filters 
# in addition, 3 max-pooling layers with size of 2x2 each will be applied as well
#  - 128.   dense layer/ fully connected layer
# -- output layer


# In[13]:


from tensorflow.keras import layers #My code


# In[16]:


# Model the Data

# first import the necessary modules required to train the model.
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU


# In[17]:


batch_size = 64
epochs = 20
num_classes = 10


# In[37]:


#Neural Network Architecture # read lecture 6

# Leaky ReLUs attempt to solve this: the function will not be zero but will instead have a small negative slope.

fashion_model = Sequential() #the original model?
fashion_model.add(Conv2D(32,kernel_size= (3,3),
                         activation = 'linear',
                         input_shape = (28,28,1), padding = 'same'))
fashion_model.add(LeakyReLU(alpha = 0.1)) #what's alpha here? = learning rate
fashion_model.add(MaxPooling2D((2,2),padding = 'same'))

fashion_model.add(Conv2D(64, (3,3),activation = 'linear',padding = 'same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

fashion_model.add(Conv2D(128, (3,3), activation = 'linear', padding = 'same'))
fashion_model.add(LeakyReLU(alpha = 0.1))
fashion_model.add(MaxPooling2D((2,2),padding = 'same'))

fashion_model.add(Flatten())

fashion_model.add(Dense(128, activation = 'linear'))
fashion_model.add(LeakyReLU(alpha = 0.1))
fashion_model.add(Dense(num_classes, activation = 'softmax'))


# In[38]:


# Compile the Model with Adam optimizer

# read cross-entropy

fashion_model.compile(loss = keras.losses.categorical_crossentropy, 
                      optimizer = keras.optimizers.Adam(),
                      metrics = ['accuracy'])


# In[39]:


#visualize the layers that you created in the above step by using the summary function. 
#This will show some parameters (weights and biases) in each layer and also the total parameters in your model.

fashion_model.summary()

#why the output shape is 4 dimensions?


# In[40]:


# Train the Model

#The model trains for 20 epochs.
#By storying the results of functions in fashion_train, 
# use it later to plot the accuracy and loss function plots 
# between training and validation which will help you to analyze your model's performance visually.

fashion_train = fashion_model.fit(train_X, train_label, batch_size = batch_size, epochs = epochs, 
                                  verbose = 1, validation_data = (valid_X, valid_label ))


# In[41]:


# Model Evaluation on the Test Set 

#Test Set for final ...? what's the theory behind this 

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose = 0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[44]:


# plot the accuracy and loss plots between training and validation data

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label = 'Training accuracy')  #bo?
plt.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show() #show?figure?


# In[18]:


# Adding Dropout into te Network

batch_size = 64
epochs = 20
num_classes = 10


# In[19]:


fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[20]:


fashion_model.summary()


# In[21]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[22]:


fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


# In[23]:


# save the model so that you can directly load it and not have to train it again for 20 epochs. This way, you can load the model later on if you need it and modify the architecture; Alternatively, you can start the training process on this saved model. It is always a good idea to save the model -and even the model's weights!- because it saves you time. Note that you can also save the model after every epoch so that, if some issue occurs that stops the training at an epoch, you will not have to start the training from the beginning.

fashion_model.save("fashion_model_dropout.h5py") 


# In[24]:


# Model Evaluation on the Test Set
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)


# In[25]:


print('Test loss:', test_eval[0])
# plot the accuracy and loss plots between training and validation dataprint('Test accuracy:', test_eval[1])


# In[27]:


# plot the accuracy and loss plots between training and validation data

accuracy = fashion_train_dropout.history['accuracy']
val_accuracy = fashion_train_dropout.history['val_accuracy']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[28]:


# Predict Labels

predicted_classes = fashion_model.predict(test_X)


# In[29]:


predicted_classes = np.argmax(np.round(predicted_classes),axis=1)


# In[30]:


predicted_classes.shape, test_Y.shape


# In[35]:


# correct label

correct = np.where(predicted_classes==test_Y)[0]

for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()


# In[36]:


# incorrect label

incorrect = np.where(predicted_classes!=test_Y)[0]

for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()


# In[37]:


# classification report 

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


# In[ ]:


# You can see that the classifier is underperforming for class 6 regarding both precision and recall. For class 0 and class 2, the classifier is lacking precision. Also, for class 4, the classifier is slightly lacking both precision and recall.

