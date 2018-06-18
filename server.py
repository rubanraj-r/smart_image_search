from flask import Flask, request, make_response, render_template

# importing the trained model
from model import CNN
import tensorflow as tf

# image preprocessing dependencies
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
    
app = Flask(__name__)
graph = tf.get_default_graph()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# return the processed image
	return image

@app.route('/predict', methods=["GET", "POST"])
def prediction():
    if request.method == 'POST':
        if request.files.get("image"):
            print('got the image - > ', request.files)
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess the image and prepare it for classification
            alt_image = prepare_image(image, target=(64, 64))
            
            # classify the input image and then initialize the list
			# of predictions to return to the client
            global graph
            with graph.as_default():
                result = model.predict(alt_image)
            
            print('result - > ', result)

            prediction = ''
            if result[0][0] == 1:
                prediction = 'dog'
            elif result[0][0] == 0:
                prediction = 'cat'
            else:
                prediction = 'other'
            print(prediction)

            return prediction

        else:
            return render_template('index.html')

if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    # load_model()
    global model
    cnn = CNN()
    model = cnn.modelCreation()
    app.run(debug=False, port=5002, host='127.0.0.1')
