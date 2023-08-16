import torch
from deep_emotion import Deep_Emotion  # Assuming `Deep_Emotion` is defined in a separate module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose the appropriate device

net = Deep_Emotion()  # Instantiate the Deep_Emotion model
net.load_state_dict(torch.load('Speaktrum_by_SOVA.pt'))  # Load the saved state dictionary
net.to(device)  # Move the model to the specified device

#------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import torch
import torch.nn.functional as F

path = "haarcascade_frontalface_default.xml"
font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not cap.isOpened():
    # Check if the webcam is opened correctly
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)

        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex+ew]  # cropping the face

            graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            final_image = cv2.resize(graytemp, (48, 48))
            final_image = np.expand_dims(final_image, axis=0)  # Add third dimension
            final_image = np.expand_dims(final_image, axis=0)  # Add fourth dimension
            final_image = final_image / 255.0  # Normalization

            data = torch.from_numpy(final_image)
            data = data.type(torch.FloatTensor)
            data = data.to(device)

            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            prediction = torch.argmax(pred)

            print(prediction)

            if (prediction == 0):
                status = "Angry, take a deep breath"
                color = (0, 0, 255)
            elif (prediction == 2):
                status = "Fear, calm down"
                color = (0, 0, 255)
            elif (prediction == 3):
                status = "Happy, you are good"
                color = (0, 0, 255)
            elif (prediction == 4):
                status = "Sad, relax and meditate"
                color = (0, 0, 255)
            else:
                status = ""
                color = (255, 0, 0)

            x1, y1, w1, h1 = 0, 0, 175, 75
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
            cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (100, 150), font, 3, color, 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    status,
                    (50, 50),
                    font, 0,
                    color,
                    2,
                    cv2.LINE_4)
        cv2.imshow('Face', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------------------


import cv2
import numpy as np
import torch
import torch.nn.functional as F

path = "haarcascade_frontalface_default.xml"
font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not cap.isOpened():
    # Check if the webcam is opened correctly
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

# Define emotions and their corresponding colors
emotions = {
    0: ("Angry, take a deep breath", (0, 0, 255)),
    2: ("Fear, calm down", (0, 0, 255)),
    3: ("Happy, you are good", (0, 255, 0)),
    4: ("Sad, relax and meditate", (255, 0, 0))
}

# Load the custom logo or watermark image
logo = cv2.imread("D:\Anaconda\Project by me\Deep-Emotion-master\logo.png", cv2.IMREAD_UNCHANGED)
logo_height, logo_width = logo.shape[:2]

while True:
    ret, frame = cap.read()

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)

        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex+ew]  # cropping the face

            graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            final_image = cv2.resize(graytemp, (48, 48))
            final_image = np.expand_dims(final_image, axis=0)  # Add third dimension
            final_image = np.expand_dims(final_image, axis=0)  # Add fourth dimension
            final_image = final_image / 255.0  # Normalization

            data = torch.from_numpy(final_image)
            data = data.type(torch.FloatTensor)
            data = data.to(device)

            outputs = net(data)
            pred = F.softmax(outputs, dim=1)
            prediction = torch.argmax(pred)

            print(prediction)

            if prediction.item() in emotions:
                emotion, color = emotions[prediction.item()]
                status = emotion
            else:
                status = "No emotion detected"
                color = (255, 255, 255)

            # Draw background rectangle for the emotion text
            bg_rect_height = 60
            bg_rect_width = int(w * 1.6)  # Adjust the width of the rectangle
            bg_rect_x = x - int((bg_rect_width - w) / 2)  # Adjust the x-coordinate of the rectangle
            bg_rect_y = y - bg_rect_height
            cv2.rectangle(frame, (bg_rect_x, bg_rect_y), (bg_rect_x + bg_rect_width, y), color, -1)


            # Draw emotion text
            text = f"Emotion: {status}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
            text_x = x + int((w - text_size[0]) / 2)
            text_y = y - bg_rect_height + int((bg_rect_height - text_size[1]) / 2)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0),1, cv2.LINE_AA)
           
        
    # Add frame to the whole screen
    frame_height, frame_width = frame.shape[:2]
    frame_with_frame = cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (255, 255, 255), 40)
    
    # Add the custom logo or watermark to the frame (top left corner)
    logo_resized = cv2.resize(logo, (int(frame_width / 4), int(frame_height / 7)))  # Resize the logo
    logo_height, logo_width = logo_resized.shape[:2]
    logo_x = 6  # Adjust the x-coordinate of the logo (distance from the left edge)
    logo_y = 6  # Adjust the y-coordinate of the logo (distance from the top edge)
    alpha_logo = logo_resized[:, :, 3] / 255.0  # Extract alpha channel
    alpha_frame = 1.0 - alpha_logo
    for c in range(3):
        frame[logo_y:logo_y + logo_height, logo_x:logo_x + logo_width, c] = (
                alpha_logo * logo_resized[:, :, c] + alpha_frame * frame[logo_y:logo_y + logo_height, logo_x:logo_x + logo_width, c]
        )        
            
            
    
    # Add frame to the whole screen
    frame_height, frame_width = frame.shape[:2]
    frame_with_frame = cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 0), 15)

    
    
    # Display the updated frame
    cv2.imshow('Speaktrum', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



