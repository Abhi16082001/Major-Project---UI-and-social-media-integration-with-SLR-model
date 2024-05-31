from django.shortcuts import render
from keras.models import model_from_json
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

json_file = open("savedModels/signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("savedModels/signlanguagedetectionmodel48x48.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def interface(request):
    return render(request, 'index.html')

def recognition(request):
        cap = cv2.VideoCapture(0)
        flag=True
        list=[" "]
        predicted_string=""
        label =['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']
        while flag:
            _,frame = cap.read()
            cv2.rectangle(frame,(0,40),(300,300),(122, 26, 14),1)
            cropframe=frame[40:300,0:300]
            cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
            cropframe = cv2.resize(cropframe,(48,48))
            cropframe = extract_features(cropframe)
            pred = model.predict(cropframe) 
            prediction_label = label[pred.argmax()]
            cv2.rectangle(frame, (0,0), (300, 40), (122, 26, 14), -1)
            if prediction_label == 'blank':
                cv2.putText(frame, " ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
                if(list[-1]!=" "):
                    list.append(" ")
                    predicted_string="".join(list)
            else:
                accu = "{:.2f}".format(np.max(pred)*100)
                cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
                if float(accu)>99.00 and list[-1]==" ":
                    list.append(prediction_label)
                    predicted_string="".join(list)
            cv2.putText(frame,predicted_string,(2,370),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255),2,cv2.LINE_AA)
            cv2.imshow("Sign Language Recognition",frame)
            cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Sign Language Recognition", 900, 200)
            cv2.setWindowProperty("Sign Language Recognition", cv2.WND_PROP_TOPMOST, 1)
            cv2.resizeWindow("Sign Language Recognition", 302, 400)

            if cv2.waitKey(250)!=-1:
                 flag=False
        cap.release()
        cv2.destroyAllWindows()
        return render(request, 'result.html',{'result':predicted_string})


