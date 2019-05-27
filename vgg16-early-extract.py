import h5py
import os, random
import numpy as np
import configparser
from keras.applications import VGG16,VGG19
from keras import models
import tensorflow as tf
from keras import layers, Model
from keras import optimizers
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


# Classify nail images as good or bent
class NailClassifier(object):
	def __init__(self, config):
		self._base_dir = config['filepaths']['base_dir']
		self._train_dir = os.path.join(self._base_dir, 'train')
		self._validation_dir = os.path.join(self._base_dir, 'val')
		self._test_dir = os.path.join(self._base_dir, 'test')
		#self._train_good_dir = os.path.join(self._train_dir, 'good')
		#self._train_bad_dir = os.path.join(self._train_dir, 'bad')
		#self._test_good_dir = os.path.join(self._test_dir,'good')
		self._test_bad_dir = os.path.join(self._test_dir,'bad')
		self._img_width, self._img_height = config.getint('inputparameters','img_width'),config.getint('inputparameters','img_height')
		self._train_size, self._validation_size, self._test_size = config.getint('inputparameters','train_size'),config.getint('inputparameters','validation_size'),config.getint('inputparameters','test_size')
		self._batch_size = config.getint('hyperparameters','batch_size')
		#self._conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(self._img_width, self._img_height, 3))
		self._conv_base = self.modelVGG16()
		self._conv_base.summary()


	def modelVGG16(self):
		#img_width, img_height = self._img_height, self._img_width

		### Build the network 
		img_input = layers.Input(shape=(self._img_width, self._img_height, 3),name='input_1')
		x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
		x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		# Block 2
		x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		#  Block 3
		x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		model_vgg16 = Model(input = img_input, output = x)

		model_vgg16.summary()
		layer_dict = dict([(layer.name, layer) for layer in model_vgg16.layers])
		del layer_dict['input_1']
		a = [layer.name for layer in model_vgg16.layers]
		for key in layer_dict.keys(): print(key)
		print ('##########################################')

		weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5' 
		# ('https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5)
		f = h5py.File(weights_path)
		for key1 in f.keys(): print(key1)
		for i in layer_dict.keys():
		    weight_names = f[i].attrs["weight_names"]
		    weights = [f[i][j] for j in weight_names]
		    index = a.index(i)
		    model_vgg16.layers[index].set_weights(weights)
		model_vgg16.summary()
		return model_vgg16
# Extract features fromm vgg
	def extractFeatures(self,directory, sample_count):
		
		#augmentation of train data
		datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
		        width_shift_range=0.2,
		        height_shift_range=0.2,
		        shear_range=0.2,
		        zoom_range=0.2,
		        horizontal_flip=True,
		        fill_mode='nearest')
		

		features = np.zeros(shape=(sample_count, 28, 28, 256))  # equal to the output of the convolutional base
		labels = np.zeros(shape=(sample_count))
		generator = datagen.flow_from_directory(directory,
	                                            target_size=(self._img_width,self._img_height),
	                                            batch_size = self._batch_size,
	                                            class_mode='binary')

		i = 0
		for inputs_batch, labels_batch in generator:
			features_batch = self._conv_base.predict(inputs_batch)
			features[i * self._batch_size: (i + 1) * self._batch_size] = features_batch
			labels[i * self._batch_size: (i + 1) * self._batch_size] = labels_batch
			i += 1
			if i * self._batch_size >= sample_count:
			    break
		return features, labels
	    
		

# Train model
	def train_model(self,config):
		epochs = 100
		train_features, train_labels = self.extractFeatures(self._train_dir, self._train_size)  # Agree with our small dataset size
		validation_features, validation_labels = self.extractFeatures(self._validation_dir, self._validation_size)
		


		# Debug
		#print(train_features)
		#print(train_labels)

		# Classifier layers
		model = models.Sequential()
		model.add(layers.Flatten(input_shape=(28,28,256)))
		#model.add(layers.Dense(256, activation='relu', input_dim=(28*28*256)))
		#model.add(layers.Dropout(rate=0.5))
		model.add(layers.Dense(256, activation='relu', input_dim=(28*28*256)))
		model.add(layers.Dropout(rate=0.5))
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dropout(rate=0.5))
		model.add(layers.Dense(1, activation='sigmoid'))
		model.summary()
		adam_opt = optimizers.Adam(float(config['hyperparameters']['learning_rate']), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

		# Compile model
		model.compile(optimizer=adam_opt,
		              loss='binary_crossentropy',
		              metrics=['acc'])

		# Train model
		history = model.fit(train_features, train_labels,
		                    epochs=epochs,
		                    batch_size=self._batch_size, 
		                    validation_data=(validation_features, validation_labels))

		# Save model
		model.save('nails_fcl_vgg16_early_extract.h5')

		


	def test(self,model_path):
		test_datagen = ImageDataGenerator(rescale=1./255)
		test_generator = test_datagen.flow_from_directory(self._test_dir,
	                                            target_size=(self._img_width,self._img_height),
	                                            batch_size = 1, class_mode=None, 
	                                            shuffle=False)
		test_features, test_labels = self.extractFeatures(self._test_dir, self._test_size)
		# Evaluate on test dataset
		model = load_model(model_path)
		score,accuracy = model.evaluate(test_features,test_labels,batch_size=self._batch_size)
		print('Test score:', score)
		print('Test accuracy:', accuracy)



# Predict result for single image
	def predict(self,model_path):
		model = load_model(model_path)
		path = random.choice([self._test_bad_dir])
		random_img = random.choice(os.listdir(path))
		img_path = os.path.join(path, random_img)
		img = image.load_img(img_path, target_size=(self._img_width, self._img_height))
		img_tensor = image.img_to_array(img) 
		img_tensor /= 255.
		#Extract features
		features = self._conv_base.predict(img_tensor.reshape(1,self._img_width, self._img_height, 3))

	    # Make prediction
		try:
			prediction = model.predict(features)
		except:
			prediction = model.predict(features.reshape(1, 28*28*256))

		# Write prediction
		if prediction < 0.5:
			print("bent nail with probability {}".format(1-prediction[0][0]))
		else:
			print("bent nail with probability {}".format(prediction[0][0]))

def main():
	#read config file
	congigfile_path = 'config.ini'
	config = configparser.ConfigParser()
	config.read(congigfile_path)
	learning_phase = config['inputparameters']['phase']
	#create instance for NailClassifier
	nail_classifier = NailClassifier(config)
	#nail_classifier.model_vgg16()
	if (learning_phase == 'train'):
		nail_classifier.train_model(config)
	elif (learning_phase == 'test'):
		nail_classifier.test('nails_fcl_vgg16_early_extract.h5')
	elif (learning_phase == 'predict'):
		nail_classifier.predict('nails_fcl_vgg16_early_extract.h5')

main()

