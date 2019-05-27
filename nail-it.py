# Python program to expose the nail classifier model as flask REST API 

# import the necessary modules 
from keras.applications import VGG16, VGG19
from keras.preprocessing.image import img_to_array 
from keras.applications import imagenet_utils 
from keras.models import load_model
from keras import optimizers
from keras import backend as K
#import tensorflow as tf 
from PIL import Image 
import numpy as np 
import flask 
import io 
import requests

# Create Flask application and initialize Keras model 
app = flask.Flask(__name__) 

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))


# Function to Load the model 
def load_my_model(): 
	
	global model	 
	model = load_model('nails_fcl_vgg16.h5')
	model._make_predict_function()
 
#	global graph 
#	graph = tf.get_default_graph() 

# Prepare the image
def prepare_image(image, target): 
	if image.mode != "RGB": 
		image = image.convert("RGB") 
	
	# Resize the image to the target dimensions 
	image = image.resize(target) 
	
	# PIL Image to Numpy array 
	image_tensor = img_to_array(image)
	image_tensor/=255.

	# return the processed image array
	return image_tensor

# Predict result
@app.route("/predict", methods =["POST" , "GET"]) 
def predict(): 
	data = {} # dictionary to store result 

# Debug	
#	data["success"] = False

	
	# Query url
	if flask.request.method == "GET":
		image_url = flask.request.args['image_url']
		down_image = requests.get(image_url)
		print (image_url)
		#image = Image.open(image_url)
		image = Image.open(io.BytesIO(down_image.content))



	# Image as file 
	if flask.request.method == "POST": 
		if flask.request.files.get("image"): 
			image = flask.request.files["image"].read() 
			image = Image.open(io.BytesIO(image)) 


	# Prepare image for the model
	image_tensor = prepare_image(image, target =(224, 224)) 
	features = conv_base.predict(image_tensor.reshape(1,224,224,3))

	# Predict results 
	#with graph.as_default():
	try:
		prediction = model.predict(features)
		print ('try') 
	except:
		prediction = model.predict(features.reshape(1,7*7*512)) 
		print ('except')

	data["predictions"] = [] 

		
	# Append prediction results
	r1 = {"label": 'bent', "probability": float(1 - prediction[0][0])} 
	r2 = {"label": 'good', "probability": float(prediction[0][0])} 
	data["predictions"].append(r1)
	data["predictions"].append(r2) 


	# return JSON response 
	return flask.jsonify(data) 



if __name__ == "__main__": 
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started")) 
	load_my_model() 
	app.run(port=3000,host='0.0.0.0')
