import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from time import sleep
import os
import fnmatch

"""Constants"""
#Loading classifiers 
face_cascade = cv2.CascadeClassifier('./data/utils/haarcascade_frontalface_default.xml')
eyePair_cascade = cv2.CascadeClassifier('./data/utils/haarcascades_haarcascade_mcs_eyepair_big.xml')
deepgaze = keras.models.load_model('./data/model/model_vastai.h5')

classification = ['center', 'down', 'down-left', 'down-right', 'left', 'right', 'up', 'up-left', 'up-right']
datasetdir = "./data/dataset/"
counter = 0
pos_prediction = 0
neg_prediction = 0

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
			roiEyes = roiColor[ex:ex + ew + 320, ey-350: ey + eh + 30]
			return roiEyes
	return frameImage

"""Processing"""
list_dir = os.listdir(datasetdir)
for directory in list_dir:
        path = "{}{}/".format(datasetdir, directory)
        print("Path = {}".format(path))
        list_imgs = os.listdir(path)
        for imgname in list_imgs:
            if(fnmatch.fnmatch(imgname, '*[!j][!p][!g]')):
                continue
            print(imgname)
            frame = cv2.imread("{}{}".format(path, imgname))

            picture = ReturnEyePairFunc(frame)
            if(picture.shape[1] == 0):
                continue
            gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_picture, (300, 135), interpolation = cv2.INTER_AREA)
            #cv2.imwrite("./data/benchmark_resized/{}".format(imgname), gray_picture)
            
            predictions = deepgaze.predict(np.array( [resized_image] ))
            #print("{}".format(predictions))

            x = predictions
            max = -999

            for i in range(9):
                if(max < x[0][i]):
                    max = x[0][i]
                    k = i
            #print("Value: {}; Class: {}.".format(min, classification[k]))
            
            """Benchmark"""
            #Center
            if ((fnmatch.fnmatch(imgname, "*_0V_0H*") or fnmatch.fnmatch(imgname, "*_0V_5H*") or fnmatch.fnmatch(imgname, "*_0V_-5H*"))
                and classification[k] == "center"):
                pos_prediction += 1
                    
            #Up
            elif ((fnmatch.fnmatch(imgname, '*_10V_0H*') or fnmatch.fnmatch(imgname, '*_10V_5H*') or fnmatch.fnmatch(imgname, '*_10V_-5H*'))
                  and classification[k] == "up"):
                pos_prediction += 1
                    
            #Down
            elif ((fnmatch.fnmatch(imgname, '*_-10V_0H*') or fnmatch.fnmatch(imgname, '*_-10V_5H*') or fnmatch.fnmatch(imgname, '*_-10V_-5H*'))
                  and classification[k] == "down"):
                pos_prediction += 1
                    
            #Left
            elif ((fnmatch.fnmatch(imgname, '*_0V_10H*') or fnmatch.fnmatch(imgname, '*_0V_15H*'))
                  and classification[k] == "left"):
                pos_prediction += 1
                    
            #Right
            elif ((fnmatch.fnmatch(imgname, '*_0V_-10H*') or fnmatch.fnmatch(imgname, '*_0V_-15H*'))
                  and classification[k] == "right"):
                pos_prediction += 1
                    
            #Up-Left
            elif ((fnmatch.fnmatch(imgname, '*_10V_10H*') or fnmatch.fnmatch(imgname, '*_10V_15H*'))
                  and classification[k] == "up-left"):
                pos_prediction += 1
                    
            #Up-Right
            elif ((fnmatch.fnmatch(imgname, '*_10V_-10H*') or fnmatch.fnmatch(imgname, '*_10V_-15H*'))
                  and classification[k] == "up-right"):
                pos_prediction += 1
                    
            #Down-Left
            elif ((fnmatch.fnmatch(imgname, '*_-10V_10H*') or fnmatch.fnmatch(imgname, '*_-10V_15H*'))
                  and classification[k] == "down-left"):
                pos_prediction += 1
                    
            #Down-Right
            elif ((fnmatch.fnmatch(imgname, '*_-10V_-10H*') or fnmatch.fnmatch(imgname, '*_-10V_-15H*'))
                  and classification[k] == "down-right"):
                pos_prediction += 1
                
            #Negative-Prediction
            else:
                neg_prediction += 1
                
################################################################
            if (fnmatch.fnmatch(imgname, "*_0V_0H*") or fnmatch.fnmatch(imgname, "*_0V_5H*") or fnmatch.fnmatch(imgname, "*_0V_-5H*")):
                cv2.putText(frame, "Actual: center", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Up
            elif (fnmatch.fnmatch(imgname, '*_10V_0H*') or fnmatch.fnmatch(imgname, '*_10V_5H*') or fnmatch.fnmatch(imgname, '*_10V_-5H*')):
                cv2.putText(frame, "Actual: up", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Down
            elif (fnmatch.fnmatch(imgname, '*_-10V_0H*') or fnmatch.fnmatch(imgname, '*_-10V_5H*') or fnmatch.fnmatch(imgname, '*_-10V_-5H*')):
                cv2.putText(frame, "Actual: down", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Left
            elif (fnmatch.fnmatch(imgname, '*_0V_10H*') or fnmatch.fnmatch(imgname, '*_0V_15H*')):
                cv2.putText(frame, "Actual: left", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Right
            elif (fnmatch.fnmatch(imgname, '*_0V_-10H*') or fnmatch.fnmatch(imgname, '*_0V_-15H*')):
                cv2.putText(frame, "Actual: right", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Up-Left
            elif (fnmatch.fnmatch(imgname, '*_10V_10H*') or fnmatch.fnmatch(imgname, '*_10V_15H*')):
                cv2.putText(frame, "Actual: up-left", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Up-Right
            elif (fnmatch.fnmatch(imgname, '*_10V_-10H*') or fnmatch.fnmatch(imgname, '*_10V_-15H*')):
                cv2.putText(frame, "Actual: up-right", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Down-Left
            elif (fnmatch.fnmatch(imgname, '*_-10V_10H*') or fnmatch.fnmatch(imgname, '*_-10V_15H*')):
                cv2.putText(frame, "Actual: down-left", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
                    
            #Down-Right
            elif (fnmatch.fnmatch(imgname, '*_-10V_-10H*') or fnmatch.fnmatch(imgname, '*_-10V_-15H*')):
                cv2.putText(frame, "Actual: down-right", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)            
################################################################

            cv2.putText(frame, "Prediction: {}".format(classification[k]), (10, 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
            cv2.putText(frame, "Positive: {}".format(pos_prediction), (185, 15), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
            cv2.putText(frame, "Negative: {}".format(neg_prediction), (185, 30), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
            cv2.imwrite("./data/benchmark/{}".format(imgname), frame)
                
            print("Positive: {}; Negative: {}".format(pos_prediction, neg_prediction))