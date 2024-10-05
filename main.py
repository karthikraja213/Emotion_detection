import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

file = open('model/MAK_model.json', 'r')
MAK_json = file.read()
file.close()

MAK_model = model_from_json(MAK_json)

MAK_model.load_weights("model/MAK_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1300, 710))
    if not ret:
        break
    detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grey_colorcode = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    heads = detect.detectMultiScale(grey_colorcode, scaleFactor=1.3, minNeighbors=5)

    
    for (x, y, w, h) in heads:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = grey_colorcode[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        
        emotion_prediction = MAK_model.predict(cropped_img)
        pos = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[pos], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Expression Reader', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
