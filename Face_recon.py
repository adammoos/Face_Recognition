import cv2
import face_recognition
import os
import numpy as np

# ---------------------------
# Load and encode known faces
# ---------------------------
path = "images"
encodeListKnown = []
classNames = []

for file in os.listdir(path):
    imgPath = os.path.join(path, file)

    try:
        img = cv2.imread(imgPath)

        if img is None:
            print(f"[SKIPPED] Cannot read {file}")
            continue

        # Convert grayscale → RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Force uint8
        img = img.astype(np.uint8)

        encodes = face_recognition.face_encodings(img)

        if len(encodes) == 0:
            print(f"[SKIPPED] No face found in {file}")
            continue

        encodeListKnown.append(encodes[0])
        classNames.append(os.path.splitext(file)[0])
        print(f"[OK] Encoded {file}")

    except Exception as e:
        print(f"[ERROR] {file}: {e}")

print("Encodage terminé.")
print("Personnes reconnues:", classNames)

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

while True:
    success, img = cap.read()
    if not success:
        continue

    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    imgSmall = imgSmall.astype(np.uint8)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            color = (0, 255, 0)
        else:
            name = "INCONNU"
            color = (0, 0, 255)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Reconnaissance faciale intelligente", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
