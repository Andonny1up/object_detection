from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

    
'''Nota: Para ejecutar con GPU Se necesita el componente de visual studio
kit de desarrollo de escritorio C++ (Desktop Development with C++), GPU NVIDIAC
con su respectivo drive y.. '''
    
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/cars.mp4")
# cap = cv2.VideoCapture("rtsp://Soporte:G0be2022*@192.168.14.249:554/streaming/channels/202/") #Soporte - Gobe2022*
# rtsp://Soporte:G0be2022*@192.168.14.249:554/user=Soporte&password=G0be2022*&channel=1
# rtsp://Soporte:G0be2022@192.168.14.249:554/cam/realmonitor?channel=1&subtype=0
cap.set(3,1280) # webcam 640 - cam 1280
cap.set(4,720)  # webcam 480 - cam 720

model = YOLO('../yolo_weigths/yolov8l.pt')

classNames = ["persona", "bicycle", "auto", "moto", "aeroplane", "bus", "train", "camion", "boat",
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

mask = cv2.imread("./mask.png")
# colores para las cajas
colours = [(255,56,56),(56,255,56),(56,56,255)]

# Tranking seguimiento
tracker = Sort(max_age=20, min_hits=3,iou_threshold=0.3) # tracking de seguimiento general
tracker_moto = Sort(max_age=20, min_hits=3,iou_threshold=0.3) # tracking de seguimiento para motos

limits = [400,297,673,297]

total_count = []
moto_count = []

while True:
    succes, img = cap.read() # captura un boleno y un frame
    img_region = cv2.bitwise_and(img,mask) # corto la region de la imagen
    
    # imagen o grafico para volverlo mas chevere
    img_graphics = cv2.imread("./graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,img_graphics,(0,0))
    
    results = model(img_region,stream=True)
    
    detections = np.empty((0,5))
    detections_moto = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            # print(x1,y1,x2,y2)
            
            # x1,y1,w,h=box.xywh[0]
            w, h = x2-x1,y2-y1
            
            
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            
            
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == "auto" or currentClass == "bus" or currentClass == "camion":
                cvzone.putTextRect(img,f'{classNames[cls]}: {conf}',(max(0,x1),max(35,y1-10)),
                               scale=1,thickness=2,offset=3,colorR=colours[0])
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),
                #                scale=1,thickness=1,offset=3)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=10,colorR=(56,56,255),colorC=(255,56,56))
                
                current_array = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,current_array))

            elif currentClass == "moto":
                cvzone.putTextRect(img,f'{classNames[cls]}: {conf}',(max(0,x1),max(35,y1-10)),
                               scale=1,thickness=2,offset=3,colorR=colours[2])
                
                current_array = np.array([x1,y1,x2,y2,conf])
                detections_moto = np.vstack((detections_moto,current_array))
                
    results_tracker = tracker.update(detections)
    results_tracker_moto = tracker_moto.update(detections_moto) 
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,255),5)
    
    for result in results_tracker:
        print("este es el resul",result)
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=10,rt=5, colorR=(255,56,56), colorC=(255,56,56))
        # Numero ID
        # cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),
        #                        scale=2,thickness=3,offset=10)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,colours[0],cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
        
        # cvzone.putTextRect(img,f'Contador: {len(total_count)}',(50,50),
        #                        scale=2,thickness=3,offset=10)
    for result in results_tracker_moto:
        print("este es el resul",result)
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=10,rt=5, colorR=colours[2], colorC=colours[2])
        # Numero ID
        # cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),
        #                        scale=2,thickness=3,offset=10)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,colours[2],cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if moto_count.count(id) == 0:
                moto_count.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
        
        # cvzone.putTextRect(img,f'Contador: {len(total_count)}',(50,50),
        #                        scale=2,thickness=3,offset=10)


    cv2.putText(img,str(len(total_count)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(255,56,56),8)
    cv2.putText(img,str(len(moto_count)),(455,100),cv2.FONT_HERSHEY_PLAIN,5,colours[2],8)
    
    cv2.imshow("Img", img)
    # cv2.imshow("ImgRegion", img_region)
    cv2.waitKey(1)
