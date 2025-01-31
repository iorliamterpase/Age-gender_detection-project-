import cv2
import numpy as np

# Read image 
image = cv2.imread('Gender-and-Age-Detection/source/girl2.jpg')
image = cv2.resize(image, (720, 640))

# Define models
face_pbtxt = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\opencv_face_detector.pbtxt"
face_pb = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\opencv_face_detector_uint8.pb"
age_prototxt = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\age_deploy.prototxt"
age_model = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\age_net.caffemodel"
gender_prototxt = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\gender_deploy.prototxt"
gender_model = r"C:\Users\HP\OneDrive\Desktop\My_Projects\python_age and gender detector\Gender-and-Age-Detection\models\gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

# Load models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gender = cv2.dnn.readNet(gender_model, gender_prototxt)

# Setup classification
age_range = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53)]
gender_classification = ['male', 'female']

# Copy image to avoid changes in image
img_cp = image.copy()

# Get image dimensions & blob
img_h = img_cp.shape[0]
img_w = img_cp.shape[1]
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

face.setInput(blob)
detected_face = face.forward()

face_bounds = []

# Draw rectangle over faces
for i in range(detected_face.shape[2]):
    confidence = detected_face[0, 0, i, 2]
    if confidence > 0.5:
        x1 = int(detected_face[0, 0, i, 3] * img_w)
        y1 = int(detected_face[0, 0, i, 4] * img_h)
        x2 = int(detected_face[0, 0, i, 5] * img_w)
        y2 = int(detected_face[0, 0, i, 6] * img_h)
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), int(round(img_h / 150)), 8)
        face_bounds.append([x1, y1, x2, y2])
if not face_bounds:
    print("No face was detected.")
    exit

for face_bound in face_bounds:
    try:
        # Extract face region with padding
        x1, y1, x2, y2 = face_bound
        face_region = img_cp[max(0, y1 - 15): min(y2 + 15, img_cp.shape[0] - 1),
                             max(0, x1 - 15): min(x2 + 15, img_cp.shape[1] - 1)]
        
        # Convert face image to blob
        blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, True)
        
        # Gender prediction
        gender.setInput(blob)
        gender_prediction = gender.forward()
        gender_result = gender_classification[gender_prediction[0].argmax()]

        # Age prediction
        age.setInput(blob)
        age_prediction = age.forward()
        age_result = age_range[age_prediction[0].argmax()]

        # Display text
        cv2.putText(img_cp, f'{gender_result}, {age_result}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

    except Exception as e:
        print(e)

# Display final image
cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
