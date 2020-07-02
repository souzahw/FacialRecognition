import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import os
import io

face_key = '300a5f6336924cfd964eae736e2792ac'
face_enpoint = 'https://faceopencv.cognitiveservices.azure.com/'
cred = CognitiveServicesCredentials(face_key)
client = FaceClient(face_enpoint,cred)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 1, False, (200,200))

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "x:" + str(w) + " y" + str(h)
        color = (0,255,0)
        storke = 1
        cv2.putText(frame, name,(x,y),font,1,color,storke,cv2.LINE_AA)

        crop_face = frame[y:y+h, x:x+w]
        ret,buf = cv2.imencode('.jpg', crop_face)
        stream = io.BytesIO(buf)
        detected_faces = client.face.detect_with_stream(stream, return_face_id=True, return_face_attributes=['age','gender','emotion'])
        ##a=1
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()