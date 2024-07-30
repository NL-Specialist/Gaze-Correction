import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(1)

# List to store clicked points
clicked_points = []
instructions = [
    "Please click on your nose tip",
    "Please click on your chin",
    "Please click on your left eye inner corner",
    "Please click on your left eye outer corner",
    "Please click on your right eye inner corner",
    "Please click on your right eye outer corner",
    "Please click on your left mouth corner",
    "Please click on your right mouth corner"
]

selected_landmarks = []
mode = 'head_pose'  # Default mode

def draw_selected_landmarks(img, face_landmarks, selected_points):
    img_h, img_w, _ = img.shape
    for idx in selected_points:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * img_w)
        y = int(landmark.y * img_h)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Draw small circle at each selected landmark point

def get_head_pose(image_points, img_w, img_h):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye inner corner
        (-225.0, 170.0, -135.0),     # Left eye outer corner
        (225.0, 170.0, -135.0),      # Right eye inner corner
        (225.0, 170.0, -135.0),      # Right eye outer corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2

def get_eye_gaze(face_landmarks, img_w, img_h):
    left_eye_inner_corner = np.array([face_landmarks.landmark[133].x * img_w, face_landmarks.landmark[133].y * img_h, face_landmarks.landmark[133].z * img_w])
    left_eye_outer_corner = np.array([face_landmarks.landmark[33].x * img_w, face_landmarks.landmark[33].y * img_h, face_landmarks.landmark[33].z * img_w])
    right_eye_inner_corner = np.array([face_landmarks.landmark[362].x * img_w, face_landmarks.landmark[362].y * img_h, face_landmarks.landmark[362].z * img_w])
    right_eye_outer_corner = np.array([face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h, face_landmarks.landmark[263].z * img_w])

    left_eye_vector = left_eye_outer_corner - left_eye_inner_corner
    right_eye_vector = right_eye_outer_corner - right_eye_inner_corner

    left_eye_direction = np.mean([left_eye_inner_corner, left_eye_outer_corner], axis=0)
    right_eye_direction = np.mean([right_eye_inner_corner, right_eye_outer_corner], axis=0)

    left_eye_up = left_eye_direction[1] < left_eye_inner_corner[1] - 10  # Adjust this threshold for more sensitivity
    right_eye_up = right_eye_direction[1] < right_eye_inner_corner[1] - 10  # Adjust this threshold for more sensitivity

    if left_eye_up and right_eye_up:
        return "at camera"
    else:
        return "away"

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 8:
        clicked_points.append((x, y))

def find_nearest_landmarks(face_landmarks, clicked_points, img_w, img_h):
    nearest_landmarks = []
    for click in clicked_points:
        min_dist = float('inf')
        nearest_landmark = None
        for i, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            dist = np.sqrt((x - click[0])**2 + (y - click[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_landmark = i
        nearest_landmarks.append(nearest_landmark)
    return nearest_landmarks

def main():
    global selected_landmarks, mode
    cv2.namedWindow('Face Mesh')
    cv2.setMouseCallback('Face Mesh', click_event)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = face_mesh.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = img.shape

        if not results.multi_face_landmarks:
            continue 

        for face_landmarks in results.multi_face_landmarks:
            if len(clicked_points) == 8 and not selected_landmarks:
                selected_landmarks = find_nearest_landmarks(face_landmarks, clicked_points, img_w, img_h)

            if selected_landmarks:
                draw_selected_landmarks(img, face_landmarks, selected_landmarks)
                image_points = np.array([[face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h] for i in selected_landmarks], dtype="double")
                
                if mode == 'head_pose':
                    p1, p2 = get_head_pose(image_points, img_w, img_h)
                    cv2.line(img, p1, p2, (255, 0, 0), 2)

                    # Determine if looking at camera or away
                    direction = "at camera" if p2[1] < p1[1] else "away"
                elif mode == 'eyes':
                    direction = get_eye_gaze(face_landmarks, img_w, img_h)  # Use eye vectors for gaze calculation

                cv2.putText(img, f"Looking: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"Mode: {mode}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Draw clicked points
                for point in clicked_points:
                    cv2.circle(img, point, 5, (0, 0, 255), -1)

                # Display instructions
                if len(clicked_points) < 8:
                    instruction = instructions[len(clicked_points)]
                    cv2.putText(img, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Face Mesh', img)
        key = cv2.waitKey(5)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):  # Switch between head pose and eyes mode
            mode = 'head_pose' if mode == 'eyes' else 'eyes'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
