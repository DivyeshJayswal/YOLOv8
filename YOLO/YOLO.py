###############################  YOLO.py ##########################################

from ultralytics import YOLO
import cv2
import numpy as np

import cv2
model = YOLO('Cubes_best.pt')  # load a custom model
camera = cv2.VideoCapture(0)
for i in range(1):
    return_value, image = camera.read()
    cv2.imwrite('project/captured/'+str(i)+'.png', image)
del(camera)


# Predict with the model on only 1 image
img=cv2.imread('project/captured/0.png')
results = model.predict(source=img,save=True, save_txt=True,project='project',name='predict/',exist_ok=True)# saves the results and saves the labels in runs/detect/predict_..
save_dir='project/predict/0.png'
#img=cv2.imread('test_images/test1.jpg')
for result in results:                                         # iterate results
    boxes = result.boxes.cpu().numpy()
    ab=[]                         # get boxes on cpu in numpy
    for box in boxes:                                          # iterate boxes
        r = box.xyxy[0].astype(int)                            # get corner points as int
        Klass = result.names[int(box.cls[0])]                  # Get Class names
        print(r,Klass)
        d=r.tolist()
        ab.append(d)
        #print(d)
    boxes = result.boxes.cpu().numpy() #Get bounding boxes as images saved
    for i, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[r[1]:r[3], r[0]:r[2]]
        cv2.imwrite('project/cropped/'+str(i) + ".png", crop)
    print("__________")
    #print(ab)  # Converts Numpy array to list.
    ##___ab are the coordinates of bbox and can be used for further calculations 
arr=np.array(ab)
print(arr[0,0])
print("____________________________________________________________")
 
