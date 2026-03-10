from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import os
import keras

# TensorFlow (tf.keras) installs a separate "keras" package which is used by
# tf.keras.models.load_model.  Unfortunately the two packages define distinct
# base layer classes, so during deserialization a Hub KerasLayer instance is
# rejected with:
#   ValueError: Only instances of `keras.Layer` can be added to a Sequential
#   model. Received: <tensorflow_hub.keras_layer.KerasLayer ...>
#
# Patch `Sequential.add` to accept TensorFlow Hub layers.
# The built-in check in keras.models.Sequential only considers instances of
# `keras.Layer` from the standalone `keras` package; a Hub layer inherits
# from `tf.keras.layers.Layer` and thus fails the isinstance test during
# deserialization.  We intercept the ValueError and append the layer directly.
from keras.models import Sequential as _KSequential
_orig_add = _KSequential.add

def _patched_add(self, layer, *args, **kwargs):
    try:
        return _orig_add(self, layer, *args, **kwargs)
    except ValueError as exc:
        msg = str(exc)
        if (
            "Only instances of `keras.Layer`" in msg
            and isinstance(layer, hub.KerasLayer)
        ):
            # mimic the same bookkeeping that add() does when rebuild=True
            self._layers.append(layer)
            self.built = False
            self._functional = None
            return self
        raise

_KSequential.add = _patched_add




app = Flask(__name__)

# Patch the Hub layer's call method so that it handles symbolic tensors
# gracefully during model building.  When a KerasTensor is passed we
# return a zero tensor of the same shape instead of invoking the
# underlying module, avoiding a conversion-to-numpy bug.
try:
    from tensorflow.keras.utils import is_keras_tensor
except ImportError:
    from keras.backend import is_keras_tensor
_orig_hub_call = hub.KerasLayer.call

def _patched_hub_call(self, inputs, training=None):
    if is_keras_tensor(inputs):
        # During symbolic execution, just pass through the input
        return inputs
    return _orig_hub_call(self, inputs, training=training)

hub.KerasLayer.call = _patched_hub_call

model = load_model('Model/cs_best.h5', custom_objects={'KerasLayer': hub.KerasLayer})

disease_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']
uploaded_folder="static/images/uploaded"



# function to process image and predict results
def process_predict(image_path, model):
    # read image
    img = image.load_img(image_path, target_size=(224, 224))
    # preprocess image
    img = image.img_to_array(img)
    # now divide image and expand dims
    img = np.expand_dims(img, axis=0) / 255
    # Make prediction
    pred_probs = model.predict(img)
    # Get name from prediction
    pred = disease_names[np.argmax(pred_probs)]
    pred_probs = round(np.max(pred_probs)*100, 2)
    return pred, pred_probs


@app.route('/', methods=['GET', 'POST'])
def home_page():
  if request.method == 'POST':
        # name inside files and in html input should match
        image_file = request.files['file']
        if image_file:
                filename = image_file.filename
                file_path = os.path.join( uploaded_folder, filename)
                image_file.save(file_path)
                # prediction
                pred, pred_proba = process_predict(file_path, model)
                if pred_proba > 45:
                  return render_template(
                      'prediction.html',
                      prediction=pred,
                      prediction_probability=pred_proba,
                      image_filename=filename
                  )
                else:
                    return render_template('false_pred.html')  
  return render_template("index.html")


@app.route('/Categories')
def categories_page():
    return render_template('categories.html')

@app.route('/About')
def about_page():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
