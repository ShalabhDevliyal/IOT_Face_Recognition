import joblib
import json
import numpy as np
import cv2
import base64
from wavelet import w2d

#private to this file only
__class_name_to_number={}
__class_number_to_name={}
__model=None

def classify_image(image_base64_data, file_path=None):
    imgs = identify_faces(file_path, image_base64_data)
    # print("hello")
    result=[]
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)

        # result.append(class_number_to_name(__model.predict(final)[0]))
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def load_save_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/final_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64):
    encoded_data = b64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def identify_faces(image_path, image_base64_data):
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if image_path:
        frame = cv2.imread(image_path)
    else:
        frame = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    cropped_faces=[]
    for i, ( x, y, w, h) in enumerate(faces):
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_Ayush():
    with open("b64.txt") as f:
        return f.read()

def class_number_to_name(num):
    return __class_number_to_name[num]

if __name__ == "__main__":
    load_save_artifacts()
    # print(classify_image(get_b64_test_image_for_Ayush(), None))
    print(classify_image(None, "D:/IOT_Face_Recognition/Unknown/face_1.png"))