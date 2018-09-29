from __future__ import print_function
from keras.datasets import mnist
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from keras.models import Model
from keras.utils import to_categorical

def Model2(input_tensor=None, train=False):
	if train:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train.reshape(x_train.shape[0],28,28,1)
		x_test = x_test.reshape(x_test.shape[0],28,28,1)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
		y_train=to_categorical(y_train,10)
		y_test = to_categorical(y_test,10)
		input_tensor = Input(shape=(28,28,1))
	elif input_tensor is None:
		raise ValueError("no tensor")
		exit()
	x = Convolution2D(6, (5,5), activation='relu', padding='same', name='block1_conv1')(input_tensor)
	x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
	x = Convolution2D(16, (5,5), activation='relu', padding='same', name='block2_conv1')(x)
	x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)
	x = Flatten(name='flatten')(x)
	x = Dense(84, activation='relu', name='fc1')(x)
	x = Dense(10, name='before_softmax')(x)
	x = Activation('softmax', name='predictions')(x)
	model = Model(input_tensor, x)
	model.name == 'model2'
	if train:
		model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
		model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, epochs=10, verbose=1)
		model.save_weights('./Model2.h5')
		score = model.evaluate(x_test, y_test, verbose=0)
		print('\n')
		print('Overall Test score:', score[0])
		print('Overall Test accuracy:', score[1])
	else:
		model.load_weights('./Model2.h5')
	return model


if __name__ == '__main__':
	Model2(train=True)
