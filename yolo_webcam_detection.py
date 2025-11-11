#!/usr/bin/env python3
"""
Real-time Object Detection with YOLO and Webcam
Python implementation using Ultralytics YOLO

Based on the tutorial by Dipankar Medhi:
https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam

This is an alternative implementation to the C# version in this repository.
"""

from ultralytics import YOLO
import cv2
import math

def main():
    """Main function to run YOLO webcam detection"""

    print("=" * 60)
    print("Real-time Object Detection with YOLO and Webcam")
    print("=" * 60)
    print("\nPress 'q' to quit\n")

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    if not cap.isOpened():
        print("ERROR: Failed to open webcam")
        return

    print("Webcam opened successfully")

    # Load YOLO model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Download automatically if not exists
    print("Model loaded successfully\n")

    # COCO dataset class names (80 classes)
    classNames = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]

    # Frame counter for FPS calculation
    frame_count = 0
    fps = 0

    print("Starting detection... Press 'q' to quit")

    # Main detection loop
    while True:
        success, img = cap.read()

        if not success:
            print("Failed to read frame from webcam")
            break

        # Run YOLO inference
        results = model(img, stream=True, verbose=False)

        # Process results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get confidence score
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Get class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Print detection info
                print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

                # Prepare label text
                label = f'{class_name} {confidence:.2f}'

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Draw label background
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (255, 0, 255),
                    -1
                )

                # Draw label text
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            fps = 30  # Approximate, should use time-based calculation

        cv2.putText(
            img,
            f'FPS: {fps}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display the frame
        cv2.imshow('YOLO Webcam Detection - Press q to quit', img)

        # Check for 'q' key to exit
        if cv2.waitKey(1) == ord('q'):
            print("\nExiting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
