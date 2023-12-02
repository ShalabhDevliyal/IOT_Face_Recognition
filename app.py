import cv2
import os

def indentify():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    capture = cv2.VideoCapture(0)

    output_folder = "D:\IOT_Face_Recognition\Images"
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for i, ( x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = frame[y:y+h, x:x+w]

            face_filename = os.path.join(output_folder, f'face_{i+1}.png')
            cv2.imwrite(face_filename, roi_color)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for( ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow('Facial Features Extraction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

indentify()