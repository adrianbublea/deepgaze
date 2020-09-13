#import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
#from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#import pickle
from pathlib import Path
import autokeras as ak

#from tensorflow.keras.models import load_model
#from IPython.display import SVG
#conda install -c anaconda ipython
#from tensorflow.python.keras.utils.vis_utils import model_to_dot

# Path to the unzipped CIFAR data
data_dir = Path("./data/dataset/")

img_height = 135
img_width = 300

image_count = len(list(data_dir.glob('**/*.jpg')))
print("# of images found:", image_count)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=1)
  # resize the image to the desired size
  print(img, [img_height, img_width])
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in train_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

features = np.array([list(x[0].numpy()) for x in list(train_ds)])
labels = np.array([x[1].numpy() for x in list(train_ds)])

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(horizontal_flip=False, vertical_flip=False, rotation_factor=False, zoom_factor=False)(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=7)
# Feed the tensorflow Dataset to the classifier.
#clf.fit(features, labels, epochs=10)

split = 5000
x_val = features[split:]
y_val = labels[split:]
x_train = features[:split]
y_train = labels[:split]

# for this example 4h should be enough
clf.fit(
    x_train,
    y_train,
    # Use your own validation set.
    validation_data=(x_val, y_val),
    epochs=100,
)

# Predict with the best model.
#predicted_y = clf.predict(x_val)
#print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_val, y_val))

# Export as a Keras Model.
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

""" try:
    model.save("model_autokeras", save_format="tf")
except: """
model.save("model_autokeras.h5")

#from tensorflow.keras.models import load_model

#os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\graphviz-2.44.1-win32\Graphviz\bin'
#model = load_model('model_autokeras.h5')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

#loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
#predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
#print(predicted_y)