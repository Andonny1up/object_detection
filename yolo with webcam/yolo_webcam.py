from ultralytics import YOLO
import cv2
import cvzone
import math
    
'''Nota: Para ejecutar con GPU Se necesita el componente de visual studio
kit de desarrollo de escritorio C++ (Desktop Development with C++)'''
    
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/motorbikes.mp4")
cap.set(3,1280) # webcam 640 - cam 1280
cap.set(4,720)  # webcam 480 - cam 720

model = YOLO('../yolo_weigths/yolov8n.pt')

classNames = ["persona", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    succes, img = cap.read()
    
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # print(x1,y1,x2,y2)
            
            # x1,y1,w,h=box.xywh[0]
            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),colorR=(56,56,255),colorC=(255,56,56))
            
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            
            
            # Class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)
            
    cv2.imshow("Img", img)
    cv2.waitKey(1)
