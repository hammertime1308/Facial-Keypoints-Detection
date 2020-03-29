from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys


if len(sys.argv) !=3 :
    print("Args are not correct")
    sys.exit()
else:
    source = sys.argv[1]
    destination = sys.argv[2]


# saved model 
model = load_model('facial.h5')

# capture frames using cv2 from source file
cap = cv2.VideoCapture(source)
ret, frame = cap.read()

# get fps and total frame count in video
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# information of video
height, width, channel = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(destination, fourcc, fps, (width, height))

frame_no = 1
print("Total frames = {}".format(total_frames))
while cap.isOpened():
    frame_no += 1
    ret, frame = cap.read()
    if frame_no <= total_frames:
        
        # identifying faces in image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        dimensions = (96, 96)
        default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(default_img, cv2.COLOR_RGB2GRAY)

        # faces in image
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        # contains all points in [x,y] format
        points = []
        
        for i,(x,y,w,h) in enumerate(faces):
            h+=10
            w+=10
            x-=5
            y-=5

            # get only face and resize image to focus on only image
            face_only = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)

            # scale to resize the predicted points
            scale_x = w/96
            scale_y = h/96

            # predict points in face_only
            face_only = face_only/255.0
            predicted =  model.predict(face_only.reshape(-1,96,96,1))

            # inverse transform for standardized values
            predicted = predicted*48+48
            predicted = predicted.reshape(-1)
            
            # get all x and y predicted co-ordinates
            predicted_x = predicted[::2]
            predicted_y = predicted[1::2]

            # scale predicted values to fit the original image
            predicted_x = (predicted_x*scale_x)+x
            predicted_y = (predicted_y*scale_y)+y

            
            # add x and y co-ordinates in (x,y) format to points list
            for px,py in zip(predicted_x,predicted_y):
                points.append([px,py])

        original = frame.copy()

        # create filled circle for all entries in points array
        # of radius 3 and white color 
        for point in points:
            cv2.circle(frame,tuple(point),3,(255,255,255),-1)

        # combine original image and image with points
        final = cv2.addWeighted(original,0,frame,1,0)

        # write the frame 
        out.write(final)                   
        
        print("Frame = {}".format(frame_no),end="\r")

    # end of conversion
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
