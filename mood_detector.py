import cv2
from deepface import DeepFace

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze mood
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        mood = result[0]['dominant_emotion']

        # Show mood on screen
        cv2.putText(frame, f"Mood: {mood}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    # Show camera   cv2.imshow("Mood Detector", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()