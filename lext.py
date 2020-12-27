import dlib,cv2
import numpy as np
from imutils import face_utils
Model_PATH = "/home/ubuntu/Desktop/lrdemo/shape_predictor_68_face_landmarks.dat"
frontalFaceDetector = dlib.get_frontal_face_detector()
l=[]

# Now the dlip shape_predictor class will take model and with the help of that, it will show 
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)

cap=cv2.VideoCapture("test.mp4")
id=0
while True:
    try:
    
        _,img=cap.read()
        print("read")
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = frontalFaceDetector(imageRGB, 0)[0]
        faceRectangleDlib = dlib.rectangle(int(face.left()),int(face.top()), int(face.right()),int(face.bottom()))
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
        
        l1={'id': id,'facial_landmarks':face_utils.shape_to_np(detectedLandmarks)}
        l.append(l1)
        id=id+1
    except Exception as e:
        print(e)
        break
        
        
dic1={'data':l}
np.savez('ltmp.npz', **dic1)
