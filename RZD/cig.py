#!pip install ultralytics
#!pip install catboost
import cv2
from PIL import Image
from ultralytics import YOLO

import imutils
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from catboost import CatBoostClassifier
cig = r'./best.pt'
points = r'./yolov8n-pose.pt'
people = r'./yolov8n.pt'
CB = r'./pointciglast.cbm'
test_img_path = './111.jpg'
model_cig = YOLO(cig)
net = YOLO(people)
net.classes = [0]
model = CatBoostClassifier(iterations=250, depth=12, learning_rate=0.1,
                           loss_function='Logloss', custom_metric=['AUC'], random_seed=42)
model.load_model(CB)
model_points = YOLO(points)
def predict_cig(model_cig=model_cig,face=net,image_file='/content/111.jpg',threshold=0.3):
  image = cv2.imread(image_file)
  face = model_points(image,verbose=False)[0]
  face_list = []
  for k in range(len(face.boxes.xyxy)):
      if int(face.boxes.cls[k])==0:
        (startX,startY,endX,endY) = (np.array(face.boxes.xyxy)[k]).astype("int")
        face_list.append((startX,startY,endX,endY,k))
  for j in face_list:
    x,y,w,h,k = j
    image1 = image[y:h,x:w]
    res = model_cig(image1,verbose=False)
    if len(res[0].boxes.xyxy)>0 and max(res[0].boxes.conf)>threshold:
      return 1
  return 0
def predict_points(model=model,model_points=model_points,img_path='/content/111.jpg'):
  results = model_points(img_path)
  result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0].flatten()
  if len(result_keypoint)!=0:
    pred = model.predict(result_keypoint)
    return pred
  return 0
ans = predict_cig(image_file='boomboom.jpg') or predict_points(img_path='boomboom.jpg')