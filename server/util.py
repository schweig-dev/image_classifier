import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def load_saved_artifacts():
    print("loading saved artifacts")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/best_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("successfully loaded saved artifacts")

def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    if imgs == []:
        print("No face detected")
        return None
    
    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        X = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': get_class_name(__model.predict(X)[0]),
            'class_probability': np.around(__model.predict_proba(X)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def filter_eye_detections(eyes):
    sorted_eyes = sorted(eyes, key=lambda x:x[1])
    filtered_eyes = sorted_eyes[:2]
    return filtered_eyes

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            filtered_eyes = filter_eye_detections(eyes)
            if len(filtered_eyes) == 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def get_class_name(class_num):
    return __class_number_to_name[class_num]


if __name__ == "__main__":
    load_saved_artifacts()

    # print(classify_image(None, "./test_images/BE1.jpg"))
    # print(classify_image(None, "./test_images/GX1.jpg"))
    print(classify_image(None, "./test_images/GX2.jpg"))
    # print(classify_image(None, "./test_images/RR1.jpg"))
    # print(classify_image(None, "./test_images/XS1.jpg"))
    # print(classify_image(None, "./test_images/XS2.png"))
    # print(classify_image(None, "./test_images/YS1.jpg"))