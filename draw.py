import cv2
import numpy as np
from keras.models import load_model
import keras
drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)


img = np.zeros((28,28,1), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    #cv2.resizeWindow('test draw',200,200)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
print(img.shape)
cv2.resize(img,(28,28))
cv2.destroyAllWindows()
print(img.shape)
model=load_model("model.h5")
img=img.reshape(1,784)
prediction=model.predict(img)
print("THE NO. ENTERED IS           {0}".format(np.argmax(prediction)))
