from keras.layers import Input, Conv1D, Dense, MaxPooling1D, Activation, \
						 Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np 
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


"""   data preparation   """

# BCG样本有4类，对各自的类别标签进行one-hot编码
def one_hot_coding(data):
	one_hot = [0., 0., 0., 0.]
	ids = []
	# print(data.shape)

	for row in data:
		# print(row)
		one_hot[int(row[0])] = 1.
		ids.append(one_hot)
		# print(ids)
		# one_hot[int(row[0])] = 0. # not work
		one_hot = [0., 0., 0., 0.]
	
	return np.concatenate((np.array(ids), data[:,1:]), axis=1)



train_data = one_hot_coding(np.loadtxt('./data/BCG4000/BCG_TRAIN.csv', delimiter=','))
test_data = one_hot_coding(np.loadtxt('./data/BCG4000/BCG_TEST.csv', delimiter=','))
X_train = train_data[:,4:]
y_train = train_data[:,:4]
X_test = test_data[:,4:]
y_test = test_data[:,:4]
# print(y_train)
# os._exit(0)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# print(y_test.shape)
# os._exit(0)


"""  build a CNN model """
# a simple one, single input & single output
model = Sequential()
# add first convolutional layer
model.add(Conv1D(filters=16, kernel_size=16, strides=1, padding='valid', input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=5))

# add 2nd convolutional layer
model.add(Conv1D(filters=32, kernel_size=16, strides=1, padding='valid'))
model.add(Activation('relu'))
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=5))

# add 3rd convolutional layer
model.add(Conv1D(filters=32, kernel_size=16, strides=1, padding='valid'))
model.add(Activation('relu'))
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=5))

model.add(Flatten())

# add 1st fully connected layer
model.add(Dense(128, activation='relu'))
# model.add(Activation('relu'))
model.add(Dropout(0.25))
# add 2nd fully connected layer
model.add(Dense(4, activation='softmax'))
# model.add(Activation('softmax'))

model.summary()

# print(samples[:, 4:, :].shape)
# print(samples[:,0:4,:].reshape((-1,4))[1100])
# print(samples[:,0:4,:].shape)
# os._exit(0)


"""    model training and predicting    """

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=80, 
		  verbose=1, callbacks=None, validation_split=0.1, shuffle=True)
score = model.evaluate(x=X_test, y=y_test, batch_size=64) # 返回compile()里指定的metrics函数值
print('testing samples num:', X_test.shape[0])
print('testing loss and accuracy:', score)
# print(model.predict(X_test))


"""    training history visualization     """ 

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
