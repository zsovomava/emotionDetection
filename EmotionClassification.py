import tensorflow as tf
from keras.api.utils import img_to_array , load_img
import numpy as np
import cv2
import os
import asyncio

model = "emotion.h5"
model = tf.keras.models.load_model(model)

batchSize = 32

#categories
trainPath = "train"
categories = os.listdir(trainPath)
categories.sort()
numOfClasses = len(categories)
emotionColor = [170, 80, 140, 40, -1, 110, 30]


#find and give back all the faces
def findFaces(grayImage):
    haarCascadeFile ="haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(haarCascadeFile)
    faces = face_cascade.detectMultiScale(grayImage)

    grayFaces = []
    facePlace = []
    for (x,y,w,h) in faces :
        grayFaces.append(grayImage[y:y+h , x:x+w])
        facePlace.append([x,y,w,h])

    return grayFaces, facePlace


#normalized the image
def prepareImagesForModel(faceImages):
    resultFaces = []
    for face in faceImages:
        resized = cv2.resize(face, (48,48), interpolation=cv2.INTER_AREA)
        imgResult = np.expand_dims(resized, axis=0)
        imgResult = imgResult / 255.0
        resultFaces.append(imgResult)
    return resultFaces

video_capture = cv2.VideoCapture(0)

def hue_shift_region(video_frame, x, y, w, h, hue_shift=60):
    # Vágd ki az arc területét
    face_region = video_frame[int(y):int(y+h), int(x):int(x+w)]
    if hue_shift >= 0:
        # Konvertálás HSV színtérbe
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # `Hue` csatorna eltolása
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # Hue csatorna 0-179 között van
        
        # Visszakonvertálás BGR színtérbe
        shifted_face = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Az átszínezett terület visszaillesztése az eredeti képbe
        video_frame[int(y):int(y+h), int(x):int(x+w)] = shifted_face

    else:
        # Konvertálás szürkeárnyalatos képbe
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Visszakonvertálás 3 csatornássá (BGR formátumba)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Világos szürke létrehozása
        light_gray = cv2.addWeighted(gray_bgr, 0.7, np.full_like(gray_bgr, 255), 0.3, 0)
        
        # Cseréld ki az eredeti régiót
        video_frame[int(y):int(y+h), int(x):int(x+w)] = light_gray


def visualized(video_frame, facePlace, emotionNumber, emotionColor):
    #print(facePlace)
    #print(emotionNumber)
    if len(facePlace):
        print(len(facePlace))
        for i in range(len(facePlace)):
            (x,y,w,h) = facePlace[i]
            hue_shift_region(video_frame, x, y, w, h, emotionColor[emotionNumber[i]])
            cv2.rectangle(video_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(video_frame, categories[emotionNumber[i]], (int(x),int(y)), font, 0.5, (209,19,77), 2)

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"
def classification(video_frame):
    tmpText = []
    grayImage = cv2.cvtColor(video_frame,cv2.COLOR_BGR2GRAY)
    faces, tmpfacePlace = findFaces(grayImage) 
    if len(faces) != 0:
        smallFaces = prepareImagesForModel(faces)
        height, width = grayImage.shape
        for i in range(len(tmpfacePlace)):
            x, y, w, h = tmpfacePlace[i]
            new_x = x - w * 0.1
            new_y = y - h * 0.1
            new_w = w + w * 0.2
            new_h = h + h * 0.2

            # Ellenőrzés, hogy az új koordináták a képen belül vannak-e
            if 0 <= new_x <= width and 0 <= new_y <= height and new_x + new_w <= width and new_y + new_h <= height:
                tmpfacePlace[i] = [new_x, new_y, new_w, new_h]
            resultArray = model.predict(smallFaces[i], verbose=1)
            #print(resultArray)
            answer = np.argmax(resultArray, axis=1)
            #print(answer)
            tmpText.append(answer[0])
            #print(tmpText[i])
        return [tmpfacePlace,tmpText]

def main():
    counter = 0
    facePlace = []
    emotionNumber = []
    tasks = []
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        if counter % 24 == 0:
            [facePlace, emotionNumber] = classification(video_frame.copy())
        visualized(video_frame, facePlace, emotionNumber, emotionColor)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        counter += 1
    video_capture.release()
    cv2.destroyAllWindows()

def saveImage(picture, facePlace, emotionNumber, emotionColor, path):
    if len(facePlace):
        for i in range(len(facePlace)):
            (x,y,w,h) = facePlace[i]
            hue_shift_region(picture, x, y, w, h, emotionColor[emotionNumber[i]])
            cv2.rectangle(picture, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(picture, categories[emotionNumber[i]], (int(x),int(y)), font, 0.5, (209,19,77), 2)
    cv2.imwrite("picture/test"+path, picture)

def testMain(path):
    picture = cv2.imread("picture/"+path)
    
    [facePlace, emotionNumber] = classification(picture.copy())
    saveImage(picture, facePlace, emotionNumber, emotionColor,path)


if __name__ == "__main__":  
    #main()
    testMain("suprised.jpg")
