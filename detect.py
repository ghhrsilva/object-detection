from ultralytics import YOLO
import cv2

# Load pre-trained model
model = YOLO("yolov8n.pt")  # lightweight model

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
