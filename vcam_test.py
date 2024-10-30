import cv2

# Replace 0 with the device index or path of your virtual camera
# For example, if vcam is set to index 1, replace 0 with 1
vcam_index = 3
cap = cv2.VideoCapture(vcam_index)

if not cap.isOpened():
    print("Error: Could not open virtual camera.")
else:
    print("Virtual camera opened successfully!")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Virtual Camera Feed', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
