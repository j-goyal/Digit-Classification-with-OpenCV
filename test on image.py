import numpy as np
import cv2
import pickle
 
########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.60 # MINIMUM PROBABILITY TO CLASSIFY

#####################################


#### LOAD THE TRAINNED MODEL 
pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)
 
#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


img = cv2.imread("7.png")
imgOriginal = img.copy()

img = np.asarray(imgOriginal)
img = cv2.resize(img,(32,32))

img = preProcessing(img)
    #cv2.imshow("Processsed Image",img)
img = img.reshape(1,32,32,1)

    #### PREDICT
classIndex = int(model.predict_classes(img))
#print(classIndex)

predictions = model.predict(img)
#print(predictions)

probVal= np.amax(predictions) # max prediciton for each class number
#print(classIndex,probVal)
 
if probVal> threshold:
    cv2.putText(imgOriginal,str(classIndex) + " "+str(probVal), (10,16), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 1)
 
cv2.imshow("Original Image",imgOriginal)
#cv2.imshow("Processed Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
