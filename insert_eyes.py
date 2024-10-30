import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Updated landmarks for left and right eyes
left_eye_landmarks = [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161]
right_eye_landmarks = [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]

def _extract_eye_region_with_alpha(frame, eye_points):
    height, width = frame.shape[:2]
    
    # Create mask for the eye region
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(eye_points)], 255)
    
    # Crop the eye region based on bounding box around eye points
    x, y, w, h = cv2.boundingRect(np.array(eye_points))
    mask_crop = mask[y:y+h, x:x+w]
    eye_crop = frame[y:y+h, x:x+w]

    # Create distance transform to calculate distances from edge inward
    gradient_mask = cv2.distanceTransform(255 - mask_crop, cv2.DIST_L2, 5)
    gradient_mask = cv2.normalize(gradient_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Stronger fade scaling to enhance the inward fade effect
    fade_mask = cv2.subtract(255, gradient_mask)
    fade_mask = np.clip(fade_mask * 2, 0, 255).astype(np.uint8)  # Further increase fade intensity

    # Apply an even larger Gaussian blur for a softer, more extended fade
    fade_mask = cv2.GaussianBlur(fade_mask, (31, 31), 0)
    fade_mask = np.where(mask_crop == 255, fade_mask, 0)  # Ensure fade stays within the eye contour

    # Apply the fade mask as the alpha channel
    eye_with_alpha = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2BGRA)
    eye_with_alpha[:, :, 3] = fade_mask  # Set the alpha with extended fade

    
    return eye_with_alpha, (x, y, w, h)

def extract_eye_region(frame, face_mesh_results):
    img_h, img_w, _ = frame.shape

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            left_eye_points = [(int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in left_eye_landmarks]
            right_eye_points = [(int(face_landmarks.landmark[idx].x * img_w), int(face_landmarks.landmark[idx].y * img_h)) for idx in right_eye_landmarks]

            left_eye_frame, left_bbox = _extract_eye_region_with_alpha(frame, left_eye_points)
            right_eye_frame, right_bbox = _extract_eye_region_with_alpha(frame, right_eye_points)

            return left_eye_frame, right_eye_frame, left_bbox, right_bbox
    return None, None, None, None

def blend_eye_region(base_image, eye_region, position):
    x, y, w, h = position
    overlay = base_image[y:y+h, x:x+w]

    # Resize eye region to match overlay dimensions if needed
    eye_region_resized = cv2.resize(eye_region, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_AREA)

    # Blend the eye region using the alpha channel
    alpha_eye = eye_region_resized[:, :, 3] / 255.0
    for c in range(3):  # Blend each color channel
        overlay[:, :, c] = overlay[:, :, c] * (1 - alpha_eye) + eye_region_resized[:, :, c] * alpha_eye

    base_image[y:y+h, x:x+w] = overlay
    return base_image

def swap_eyes(image1, image2):
    results1 = face_mesh.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    results2 = face_mesh.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    left_eye1, right_eye1, left_bbox1, right_bbox1 = extract_eye_region(image1, results1)
    left_eye2, right_eye2, left_bbox2, right_bbox2 = extract_eye_region(image2, results2)

    if left_eye1 is not None and right_eye1 is not None:
        # Swap left eyes
        # image1 = blend_eye_region(image1, left_eye2, left_bbox1)
        image2 = blend_eye_region(image2, left_eye1, left_bbox2)

        # Swap right eyes
        # image1 = blend_eye_region(image1, right_eye2, right_bbox1)
        image2 = blend_eye_region(image2, right_eye1, right_bbox2)

    return image1, image2

# Load images
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')

# Swap eyes
output_image1, output_image2 = swap_eyes(image1, image2)

cv2.imwrite('output_image2.png', output_image2)