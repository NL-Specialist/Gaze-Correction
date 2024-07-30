import cv2
import mediapipe as mp
import numpy as np

############## PARAMETERS #######################################################

# Show or hide certain vectors of the estimation
DRAW_GAZE = True
DRAW_FULL_AXIS = True
DRAW_HEADPOSE = False

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
X_SCORE_MULTIPLIER = 1
Y_SCORE_MULTIPLIER = 1

# Threshold of how close scores should be to average between frames
THRESHOLD = 0.3

#################################################################################

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(1)

# 3D Model points of facial landmarks (for head pose estimation)
FACE_3D = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left Mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
], dtype=np.float64)

# Initialize gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

def reposition_face(face_3d):
    """Reposition the eye corners to be the origin for left and right eyes."""
    leye_3d = np.array(face_3d)
    leye_3d[:, 0] += 225
    leye_3d[:, 1] -= 175
    leye_3d[:, 2] += 135

    reye_3d = np.array(face_3d)
    reye_3d[:, 0] -= 225
    reye_3d[:, 1] -= 175
    reye_3d[:, 2] += 135

    return leye_3d, reye_3d

def calculate_gaze_score(face_2d, last_score, idx1, idx2, idx3, threshold):
    """Calculate the gaze score for a given set of indices."""
    if (face_2d[idx2][0] - face_2d[idx1][0]) != 0:
        score = (face_2d[idx3][0] - face_2d[idx1][0]) / (face_2d[idx2][0] - face_2d[idx1][0])
        if abs(score - last_score) < threshold:
            score = (score + last_score) / 2
        return score
    return last_score

def calculate_head_pose(leye_3d, reye_3d, face_2d_head, cam_matrix, dist_coeffs):
    """Calculate the head pose for left and right eyes."""
    _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return l_rvec, l_tvec, r_rvec, r_tvec

def draw_axis(img, corner, axis, rvec, tvec, cam_matrix, dist_coeffs, color):
    """Project and draw the axis of rotation."""
    axis_points, _ = cv2.projectPoints(axis, rvec, tvec, cam_matrix, dist_coeffs)
    cv2.line(img, corner, tuple(np.ravel(axis_points[0]).astype(np.int32)), color, 3)
    cv2.line(img, corner, tuple(np.ravel(axis_points[1]).astype(np.int32)), color, 3)
    cv2.line(img, corner, tuple(np.ravel(axis_points[2]).astype(np.int32)), color, 3)

def main():
    leye_3d, reye_3d = reposition_face(FACE_3D)

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
            face_2d = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in face_landmarks.landmark]

            face_2d_head = np.array([
                face_2d[1],    # Nose
                face_2d[199],  # Chin
                face_2d[33],   # Left eye left corner
                face_2d[263],  # Right eye right corner
                face_2d[61],   # Left mouth corner
                face_2d[291]   # Right mouth corner
            ], dtype=np.float64)

            global last_lx, last_rx, last_ly, last_ry
            lx_score = calculate_gaze_score(face_2d, last_lx, 130, 243, 468, THRESHOLD)
            last_lx = lx_score

            ly_score = calculate_gaze_score(face_2d, last_ly, 27, 23, 468, THRESHOLD)
            last_ly = ly_score

            rx_score = calculate_gaze_score(face_2d, last_rx, 463, 359, 473, THRESHOLD)
            last_rx = rx_score

            ry_score = calculate_gaze_score(face_2d, last_ry, 257, 253, 473, THRESHOLD)
            last_ry = ry_score

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            l_rvec, l_tvec, r_rvec, r_tvec = calculate_head_pose(leye_3d, reye_3d, face_2d_head, cam_matrix, dist_coeffs)

            axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)

            l_corner = face_2d_head[2].astype(np.int32)
            if DRAW_HEADPOSE:
                draw_axis(img, l_corner, axis, l_rvec, l_tvec, cam_matrix, dist_coeffs, (200, 200, 0))
            if DRAW_GAZE:
                l_gaze_rvec = np.array(l_rvec)
                l_gaze_rvec[2][0] -= (lx_score - 0.5) * X_SCORE_MULTIPLIER
                l_gaze_rvec[0][0] += (ly_score - 0.5) * Y_SCORE_MULTIPLIER
                draw_axis(img, l_corner, axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs, (255, 0, 0))

            r_corner = face_2d_head[3].astype(np.int32)
            if DRAW_HEADPOSE:
                draw_axis(img, r_corner, axis, r_rvec, r_tvec, cam_matrix, dist_coeffs, (200, 200, 0))
            if DRAW_GAZE:
                r_gaze_rvec = np.array(r_rvec)
                r_gaze_rvec[2][0] -= (rx_score - 0.5) * X_SCORE_MULTIPLIER
                r_gaze_rvec[0][0] += (ry_score - 0.5) * Y_SCORE_MULTIPLIER
                draw_axis(img, r_corner, axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs, (255, 0, 0))

        cv2.imshow('Head Pose Estimation', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
