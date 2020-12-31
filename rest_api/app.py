from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(32, 32))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            results = temp = zip([label_list[i] for i in np.argsort(preds[0])][::-1],
                                    list(preds[0][np.argsort(preds[0])][::-1]))
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (label, prob) in results:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

def load_model():
    global model
    model = keras.models.load_model('assets/cifar10_model.h5')
    print('Model loaded')

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # same preprocessing steps that training data underwent
    image = (image.astype('float32') - 120.70756512369792) / (64.06097012299574+1e-9)

    # return the processed image
    return image

if __name__ == "__main__":
    load_model()
    app.run(debug=True)
