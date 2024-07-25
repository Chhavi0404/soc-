import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
#from keras.optimizers import Adam
#from keras.metrics import categorical_crossentropy

(train_image, train_label),(test_image,test_label) = keras.datasets.mnist.load_data()
train_image = train_image.astype('float32')/255
test_image = test_image.astype('float32')/255
train_label= keras.utils.to_categorical(train_label)
test_label= keras.utils.to_categorical(test_label)

model = models.Sequential()
model.add(layers.Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(layers.Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=2))
model.add(layers.Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(392,activation='relu'))
model.add(layers.Dropout(.25))
model.add(layers.Dense(124,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

model.compile(optimizer,loss, metrics=['accuracy'])

model.fit(train_image,train_label,epochs=5,batch_size=64)
score = model.evaluate(test_image,test_label)

model.save("model1.h5")