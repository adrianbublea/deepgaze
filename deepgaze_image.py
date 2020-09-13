import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from time import sleep

"""Constants"""
#Loading classifiers 
face_cascade = cv2.CascadeClassifier('./data/utils/haarcascade_frontalface_default.xml')
eyePair_cascade = cv2.CascadeClassifier('./data/utils/haarcascades_haarcascade_mcs_eyepair_big.xml')
deepgaze = keras.models.load_model('./data/model/model_vastai.h5')

classification = ['center', 'down', 'down-left', 'down-right', 'left', 'right', 'up', 'up-left', 'up-right']

# #For FPS
# lastFrameTime = time.time()

"""Functions"""
#Eye cropping function
def ReturnEyePairFunc(frameImage):
	gray = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for x, y, w, h in faces: # face points
		roiGray = gray[y:y + h, x:x + w]
		roiColor = frameImage[y:y + h, x:x + w]
		eyePairs = eyePair_cascade.detectMultiScale(roiGray)

		for (ex, ey, ew, eh) in eyePairs: # eye_pair points
			roiEyes = roiColor[ex:ex + ew + 120, ey-90: ey + eh + 30]
			return roiEyes
	return frameImage

"""Processing"""
"""
    The feeded images needs to be gray.
    If not, convert them to gray first!
"""
frame = cv2.imread("./data/dataset/down-left/0001_2m_0P_-10V_10H.jpg")
picture = ReturnEyePairFunc(frame)
gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_picture, (300, 135), interpolation = cv2.INTER_AREA)

predictions = deepgaze.predict(np.array( [resized_image] ))
#print("{}".format(predictions))

x = predictions
max = -999

for i in range(9):
    if(max < x[0][i]):
        max = x[0][i]
        k = i
#print("Value: {}; Class: {}.".format(min, classification[k]))
cv2.putText(frame, "Prediction: {}".format(classification[k]), (10, 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
cv2.imwrite("image.jpg", frame)
print("Image written to main directory.")