import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image

class FaceModel:
	def __init__(self):
		self.dataset = pd.read_csv('emotions_dataset.csv')
		self.features = 64
		self.labels = 7 
		self.batch_size = 64
		self.epochs = 40
		self.width = 48
		self.height = 48

	def model(self):
		self.X_train, self.Y_train, self.X_test, self.Y_test = [], [], [], []

		for idx, val in self.dataset.iterrows():
			self.value = val['pixels'].split(' ')
			try:
				if 'Training' in val['Usage']:
					self.X_train.append(np.array(self.value, 'float32'))
					self.Y_train.append(val['emotion'])
				elif 'PublicTest' in val['Usage']:
					self.X_test.append(np.array(self.value, 'float32'))
					self.Y_test.append(val['emotion'])
			except: 
				print ('Error in creating training and testing dataset')

		self.X_train, self.Y_train = np.array(self.X_train, 'float32'), np.array(self.Y_train, 'float32')
		self.X_test, self.Y_test = np.array(self.X_test, 'float32'), np.array(self.Y_test, 'float32')
		self.Y_train, self.Y_test = np_utils.to_categorical(self.Y_train, num_classes = self.labels), np_utils.to_categorical(self.Y_test, num_classes = self.labels)

		self.X_train -= np.mean(self.X_train, axis = 0)
		self.X_test -= np.mean(self.X_test, axis = 0)

		self.X_train /= np.std(self.X_train, axis = 0)
		self.X_test /= np.std(self.X_test, axis = 0)

		self.X_train = self.X_train.reshape(self.X_train.shape[0], self.width, self.height, 1)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], self.width, self.height, 1)

		self.model = Sequential()
		self.model.add(Conv2D(64, kernel_size = (3, 3), activation ='relu', input_shape = (self.X_train.shape[1:])))
		self.model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
		self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
		self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
		self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Conv2D(128, (3, 3), activation = 'relu'))
		self.model.add(Conv2D(128, (3, 3), activation = 'relu'))
		
		self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
		self.model.add(Flatten())

		self.model.add(Dense(1024, activation = 'relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1024, activation = 'relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(7, activation = 'softmax'))

		self.model.compile(loss = categorical_crossentropy, optimizer = Adam(), metrics = ['accuracy'])
		self.model.fit(self.X_train, self.Y_train, batch_size = self.batch_size, epochs = self.epochs, verbose = 1, validation_data = (self.X_test, self.Y_test), shuffle = True)

		self.save_model = self.model.to_json()
		with open('classifier.json', 'w') as jFile:
			jFile.write(self.save_model)
		self.model.save_weights('classifier.h5')

class FaceRecognition:

	def __init__(self, cascade):
		self.cascade = cv2.CascadeClassifier(cascade)
		self.capture = cv2.VideoCapture(0)
		self.feelings = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

	def run(self):

		with open('classifier.json', 'r') as f:
			self.model = model_from_json(f.read())
			self.model.load_weights('classifier.h5')
			
			while (True):
				self.retrieve, self.frame = self.capture.read()
				self.gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
				self.detected = self.cascade.detectMultiScale(self.gray_image, scaleFactor = 1.5,  minNeighbors = 5)
				
				for (x, y, h, w) in self.detected:
					cv2.rectangle(self.frame, (x, y), (x + w, y + h), (200, 0, 0), thickness = 5)
					self.crop_gray = self.gray_image[y : y + h, x : x + w]
					self.crop_gray = cv2.resize(self.crop_gray, (48, 48))
					self.pixels = image.img_to_array(self.crop_gray)
					self.pixels = np.expand_dims(self.pixels, axis = 0)
					self.pixels /= 255

					self.y_prediction = self.model.predict(self.pixels)
					self.emotion = self.feelings[np.argmax(self.y_prediction[0])]
					cv2.putText(self.frame, self.emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)


				cv2.imshow('frame', self.frame)
				if cv2.waitKey(20) & 0xFF == ord('q'):
					break

			self.capture.release()
			cv2.destroyAllWindows()

# Training the emotions model
# DO NOT activate until you have 5 hours in spare
# FaceModel().model()

FaceRecognition(cascade = 'haarcascade_frontalface_default.xml').run()