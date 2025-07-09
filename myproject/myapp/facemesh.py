import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# New logic variables
face_absent_start = None
face_absent_count = 0
face_present = True

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Check if face is detected
    if results.multi_face_landmarks:
        face_present = True
        face_absent_start = None  # reset the timer since face is present

        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
    else:
        if face_present:  # just lost the face
            face_absent_start = time.time()
            face_present = False
        elif face_absent_start is not None:
            elapsed = time.time() - face_absent_start
            if elapsed >= 4:
                face_absent_count += 1
                print(f"[!] Face missing for {int(elapsed)}s. Count: {face_absent_count}")
                face_absent_start = None  # reset timer after counting

    # Exit if face absent 3 times (≥4s each)
    if face_absent_count >= 3:
        print("No face detected 3 times. Exiting...")
        break

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
