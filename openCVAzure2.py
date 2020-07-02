import cv2
import os
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

face_key = '300a5f6336924cfd964eae736e2792ac'
face_endpoint = 'https://faceopencv.cognitiveservices.azure.com/'
credentials = CognitiveServicesCredentials(face_key)
client = FaceClient(face_endpoint, credentials)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(cv2.data.haarcascades)
video_capture = cv2.VideoCapture(0)
csfinished = False
emotions = []
framecount = 0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 1, False, (200, 200))
    if faces is None:
        csfinished == False

    #Desenhar retangulo envolta da face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        crop_img = frame[y:y + h, x:x + w]
        # cv2.imshow("cropped", crop_img)
        if framecount == 0:
            ret1, buf1 = cv2.imencode('.jpg', crop_img)
            stream = io.BytesIO(buf1)
            detected_faces = client.face.detect_with_stream(
                stream,
                return_face_id=True,
                return_face_attributes=['age', 'gender', 'emotion'])
            csfinished = True
            emotionfound = False
            for detected_face in detected_faces:
                label = str(detected_face.face_attributes.age)
                emotions = detected_face.face_attributes.emotion
                highemotionScore = emotions.anger
                highemotionName = "Raiva"
                if emotions.contempt > highemotionScore:
                    highemotionScore = emotions.contempt
                    highemotionName = "Desprezo"
                if emotions.disgust > highemotionScore:
                    highemotionScore = emotions.disgust
                    highemotionName = "Nojo"
                if emotions.fear > highemotionScore:
                    highemotionScore = emotions.fear
                    highemotionName = "Medo"
                if emotions.happiness > highemotionScore:
                    highemotionScore = emotions.happiness
                    highemotionName = "Feliz"
                if emotions.neutral > highemotionScore:
                    highemotionScore = emotions.neutral
                    highemotionName = "Neutro"
                if emotions.sadness > highemotionScore:
                    highemotionScore = emotions.sadness
                    highemotionName = "Triste"
                if emotions.surprise > highemotionScore:
                    highemotionScore = emotions.surprise
                    highemotionName = "Surpreso"
                if highemotionScore > 0:
                    emotionfound = True
        framecount += 1
        if framecount == 30:
            framecount = 0
        if emotionfound:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = highemotionName + " " + str(highemotionScore)
            color = (0, 255, 0)
            storke = 1
            cv2.putText(frame, name, (x, y), font, 1, color, storke, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()