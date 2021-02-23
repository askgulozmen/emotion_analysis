import cv2
import numpy as np
import os
import tensorflow as tf
import Face_Detection
import Face_Detection2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

veri_konumu = 'C:\\Users\\Ayse\\Desktop\\ayse_proje\\Face_Exp_INC2\\'


categoriler = os.listdir(veri_konumu)
print(categoriler)

model = tf.keras.models.load_model(r"C:\Users\Ayse\Desktop\ayse_proje\project_files\Facial_Exp_v0.1")

cap = cv2.VideoCapture(0)


while True:
    ret, cerceve = cap.read()

    faces = Face_Detection2.detect_face(cerceve)
    if ret:
        for sep in faces:
            image = cerceve[sep[1]-40:sep[3]+30, sep[0]:sep[2]]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (60, 60))
            gray = gray.reshape(-1, 60, 60, 1)
            prediction = model.predict([gray])
            prediction = prediction.tolist()
            max_value_of_prediction = max(prediction[0])
            our_prediction = prediction[0].index(max_value_of_prediction)
            kategori = categoriler[our_prediction]


            image_2 = cv2.rectangle(cerceve, (sep[0], sep[1]-40), (sep[2], sep[3]+30), (0, 255, 255), 2)
            image_2 = cv2.putText(image_2, kategori, (sep[2]-100, sep[3]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            cv2.imshow("Image_2", image_2)
            #cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
