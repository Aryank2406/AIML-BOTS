import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")


# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Load the video stream

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the video stream.")
    sys.exit()


while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
   

if not ret:
    print("Error: Unable to read frame from the video stream.")
    sys.exit()

if frame is None:
    print("Error: Empty frame.")
    sys.exit()


    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Determine the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Compute the eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio for both eyes v
        ear = (left_ear + right_ear) / 2.0

    # Check if the eye aspect ratio is below a certain threshold (indicating closed eyes)
    if ear < 0.25:
         cv2.putText(frame, "Sleeping", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            


    # Display the frame
    cv2.imshow("Frame", frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Break the loop if 'q' is pressed
  #  if cv2.waitKey(1) & 0xFF == ord('q'):
   #     break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
