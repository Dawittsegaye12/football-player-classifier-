import cv2
import numpy as np
import json
import joblib
import base64

from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(image_base64_data,file_path=None):

    imgs = get_cropped_images_if_2_eyes(file_path,image_base64_data)

    result = []

    for img in imgs:

        scaled_img = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scaled_img_har = cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scaled_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))

        len_img_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_img_array).astype(float)
        probability_list = __model.predict_proba(final)[0].tolist()
        class_ = class_number_to_name(probability_list.index(max(probability_list)))

        result.append({
            'class': class_,
            'class_probability': np.round(__model.predict_proba(final) * 100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

# def classify_image(image_base64_data, file_path=None):

#     imgs = get_cropped_images_if_2_eyes(file_path, image_base64_data)

#     result = []
#     for img in imgs:
#         scalled_raw_img = cv2.resize(img, (32, 32))
#         img_har = w2d(img, 'db1', 5)
#         scalled_img_har = cv2.resize(img_har, (32, 32))
#         combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

#         len_image_array = 32*32*3 + 32*32

#         final = combined_img.reshape(1,len_image_array).astype(float)
#         result.append({
#             'class': class_number_to_name(__model.predict(final)[0]),
#             'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
#             'class_dictionary': __class_name_to_number
#         })

#     return result

def get_cropped_images_if_2_eyes(img_path,img_base64_data):

    face_cascade = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_eye.xml')

    if img_path:
        img = cv2.imread(img_path)

    else:
        img = get_cv2_img_from_base64_img(img_base64_data)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    cropped_faces = []

    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if (len(eyes) >= 2):
            cropped_faces.append(roi_color)

    return cropped_faces

def get_cv2_img_from_base64_img(base64_image_data):

    encoded_data = base64_image_data.split(',')[1]

    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    return img

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def load_saved_artifacts():

    print('Loading artifacts started......')

    global __class_number_to_name
    global __class_name_to_number
    global __model

    with open('./artifacts/class_dictionary.json','r') as f:

        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    with open('./artifacts/best_model.pkl','rb') as f:

        __model = joblib.load(f)

def check_base64_img_for_ronaldo():

    with open('ronaldo_b64.txt') as f:
        return f.read()


if __name__ == '__main__':

    load_saved_artifacts()
    print(classify_image(check_base64_img_for_ronaldo(),None))
    print(classify_image(None,'../test_images/cr7.jpg'))
    print(classify_image(None,'../test_images/henry.png'))
    print(classify_image(None,'../test_images/messi.jpeg'))
    print(classify_image(None,'../test_images/ronaldinho.png'))
    print(classify_image(None,'../test_images/vieira.png'))

