import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from time import time

"""Constants"""
#Loading classifiers 
face_cascade = cv2.CascadeClassifier('./data/utils/haarcascade_frontalface_default.xml')
eyePair_cascade = cv2.CascadeClassifier('./data/utils/haarcascades_haarcascade_mcs_eyepair_big.xml')
deepgaze = keras.models.load_model('./data/model/model_vastai.h5')

classification = ['center', 'down', 'down-left', 'down-right', 'left', 'right', 'up', 'up-left', 'up-right']

# #For FPS
lastFrameTime = time()

#Video imput
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
			roiEyes = roiColor[ex:ex + ew + 35, ey - 90 - 150: ey + eh + 30]
			return roiEyes
	return frameImage

"""Processing"""
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    #Processing the image
    picture = ReturnEyePairFunc(frame)
    if(picture.shape[1] == 0):
        continue
    gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_picture, (300, 135), interpolation = cv2.INTER_AREA)
    #cv2.imshow("Eyes", resized_image)
    #Applying the prediction
    predictions = deepgaze.predict(np.array( [resized_image] ))
    #print("{}".format(predictions))
    
    #Getting the prediction (highest probability value)
    x = predictions
    max = -999
    for i in range(9):
        if(max < x[0][i]):
            max = x[0][i]
            k = i
    #print("Value: {};      Class: {}.\n".format(min, classification[k]))
    
    #FPS text
    cv2.putText(frame, "FPS: %.2f" %(1/(time() - lastFrameTime)), (530, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    lastFrameTime = time() 
    #Prediction text
    cv2.putText(frame, "Direction: {}".format(classification[k]), (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    
    #Show frame
    cv2.imshow("Frame", frame)

    #ESC key will stop the processing
    key = cv2.waitKey(1)
    if key == 27:
        break
        
# Cleanup the cap and close any open windows
cap.release()
cv2.destroyAllWindows()