import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils

# Loading MNIST Dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

# Finding No. of Rows and Columns
rows_of_img = x_train[0].shape[0]
cols_of_img = x_train[1].shape[0]

# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)
x_train = x_train.reshape(x_train.shape[0], rows_of_img, cols_of_img, 1)
x_test = x_test.reshape(x_test.shape[0], rows_of_img, cols_of_img, 1)

# store the shape of a single image 
input_shape = (rows_of_img, cols_of_img, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Featuring Scaling - Normalization
x_train /= 255
x_test /= 255

# Doing One-Hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

n_classes = y_test.shape[1]


# In[17]:

# Set Kernel Size
value = int(sys.argv[1])
kernel_size = (value,value)

# Creating model
model = Sequential()

# Adding CRP layers
model.add(Conv2D(20,kernel_size,padding="same",input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(50,kernel_size,padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# FC
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

print(model.summary())


# In[ ]:

'''
# Training Parameters
batch_size = 128
epochs = 10

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("mnist_LeNet.h5")

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


'''



# In[ ]:
from os import system
accuracy = 10
#system("bash -c echo '{}' | cat > /root/1-Pull-Code/result".format(accuracy))
file = open("result.txt", "w+")
file.write(str(accuracy))
file.close()
