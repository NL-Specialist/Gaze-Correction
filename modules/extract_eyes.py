import cv2
import numpy as np
from modules.eyes import Eyes

# Function to load an image from a file
def load_image(path):
    return cv2.imread(path)

# Function to process face landmarks in an image
def process_face_landmarks(image, eyes_processor):
    # Convert image to RGB and process face landmarks
    img_h, img_w, _ = image.shape
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = eyes_processor.face_mesh.process(frame_rgb)
    return results, img_h, img_w

# Function to draw contours around the eyes
def draw_eye_contours(image, results, img_h, img_w, thickness):
    # Initialize lists to store eye points
    left_eye_points = []
    right_eye_points = []

    # Iterate over face landmarks
    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)

            # Check if landmark is part of the left or right eye
            if idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173]:
                left_eye_points.append((x, y))
            elif idx in [362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                right_eye_points.append((x, y))

    # Draw contours around the eyes
    draw_contours(image, left_eye_points,right_eye_points, (0, 255, 0), thickness)
    # draw_contours(image, right_eye_points, (0, 255, 0))

# Function to draw contours between points
def draw_contours(image, left_eye_points, right_eye_points, color, thickness):
    # Draw the green lines around the left eye in the required order
    if len(left_eye_points) > 0:
        cv2.line(image, left_eye_points[0], left_eye_points[12], color, thickness)
        cv2.line(image, left_eye_points[0], left_eye_points[11], color, thickness)
        cv2.line(image, left_eye_points[11], left_eye_points[10], color,thickness)
        cv2.line(image, left_eye_points[10], left_eye_points[9], color, thickness)
        cv2.line(image, left_eye_points[9], left_eye_points[6], color,  thickness)
        # cv2.line(image, left_eye_points[8], left_eye_points[7], color,  thickness)
        # cv2.line(image, left_eye_points[7], left_eye_points[6], color,  thickness)
        cv2.line(image, left_eye_points[6], left_eye_points[5], color,  thickness)
        cv2.line(image, left_eye_points[5], left_eye_points[4], color,  thickness)
        cv2.line(image, left_eye_points[4], left_eye_points[3], color,  thickness)
        cv2.line(image, left_eye_points[3], left_eye_points[2], color,  thickness)
        cv2.line(image, left_eye_points[2], left_eye_points[12], color, thickness)

    # Draw the green lines around the right eye in the required order
    if len(right_eye_points) > 0:
        cv2.line(image, right_eye_points[0], right_eye_points[10], color, thickness)
        cv2.line(image, right_eye_points[10], right_eye_points[4], color, thickness)
        cv2.line(image, right_eye_points[4], right_eye_points[5], color,  thickness)
        cv2.line(image, right_eye_points[5], right_eye_points[6], color,  thickness)
        # cv2.line(image, right_eye_points[6], right_eye_points[7], color,  thickness)
        # cv2.line(image, right_eye_points[7], right_eye_points[8], color,  thickness)
        cv2.line(image, right_eye_points[6], right_eye_points[11], color, thickness)
        cv2.line(image, right_eye_points[11], right_eye_points[9], color, thickness)
        cv2.line(image, right_eye_points[9], right_eye_points[1], color,  thickness)
        cv2.line(image, right_eye_points[1], right_eye_points[2], color,  thickness)
        cv2.line(image, right_eye_points[2], right_eye_points[3], color,  thickness)
        cv2.line(image, right_eye_points[3], right_eye_points[0], color,  thickness)

# Function to detect green contours with positive or negative padding
def detect_green_contours(image, padding=1):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a very strict HSV range for the color green (0, 255, 0)
    lower_green = np.array([35, 150, 150])  # Slightly broader range to capture close colors
    upper_green = np.array([85, 255, 255])  # Slightly broader range to capture close colors
    
    # Create a mask to detect green
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours from the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for padding
    padding_mask = np.zeros_like(green_mask)
    
    # Draw contours on the padding mask
    for contour in contours:
        cv2.drawContours(padding_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Choose dilation or erosion based on the padding value
    kernel_size = max(abs(padding) * 2 + 1, 3)  # Ensure kernel is at least 3x3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if padding > 0:
        # Apply dilation for positive padding
        padded_mask = cv2.dilate(padding_mask, kernel, iterations=1)
    else:
        # Apply erosion for negative padding (shrink contours)
        padded_mask = cv2.erode(padding_mask, kernel, iterations=1)
    
    # Find new contours from the padded mask
    padded_contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return padded_contours, green_mask

# Main function
def extract(image_path='my_frame.jpg', dest_image_path='sam_test_image_away.jpg'):
    # Load input and destination images
    image = load_image(image_path)
    copy_image = load_image(image_path)
    dest_image = load_image(dest_image_path)
    copy_dest_image = load_image(dest_image_path)

    # Initialize Eyes class
    eyes_processor = Eyes()
    dest_eyes_processor = Eyes()

    # Process face landmarks in input image
    results, img_h, img_w = process_face_landmarks(image, eyes_processor)
    draw_eye_contours(image, results, img_h, img_w, 1)

    # Process face landmarks in destination image
    dest_results, dest_img_h, dest_img_w = process_face_landmarks(dest_image, dest_eyes_processor)
    draw_eye_contours(dest_image, dest_results, dest_img_h, dest_img_w, 1)

    
    
    # Detect green contours and mask
    contours, green_mask = detect_green_contours(image,1)
    dest_contours, dest_green_mask = detect_green_contours(dest_image,1)


    # Extract the eye regions from the input image
    left_eye_contour = contours[0]
    right_eye_contour = contours[1]

    # Get the contours for the eyes in the destination image
    dest_left_eye_contour = dest_contours[0]
    dest_right_eye_contour = dest_contours[1]

    # Create masks for left and right eye regions
    left_eye_mask = np.zeros_like(copy_image)  # Create an all-black mask
    cv2.drawContours(left_eye_mask, [left_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)  # Draw filled contour in white

    right_eye_mask = np.zeros_like(copy_image)  # Create an all-black mask
    cv2.drawContours(right_eye_mask, [right_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)  # Draw filled contour in white

    # Enlarge masks
    kernel = np.ones((3, 3), np.uint8)
    left_eye_mask = cv2.dilate(left_eye_mask, kernel, iterations=2)
    right_eye_mask = cv2.dilate(right_eye_mask, kernel, iterations=2)

    # Extract eye regions using enlarged masks
    left_eye_region = cv2.bitwise_and(copy_image, left_eye_mask)
    right_eye_region = cv2.bitwise_and(copy_image, right_eye_mask)

    # Define the points for the perspective transform
    left_eye_points = left_eye_contour.reshape(-1, 2).astype(np.float32)
    left_eye_points = np.float32([left_eye_points.min(axis=0), [left_eye_points[:, 0].max(), left_eye_points[:, 1].min()], 
                                left_eye_points.max(axis=0), [left_eye_points[:, 0].min(), left_eye_points[:, 1].max()]])
    dest_left_eye_points = dest_left_eye_contour.reshape(-1, 2).astype(np.float32)
    dest_left_eye_points = np.float32([dest_left_eye_points.min(axis=0), [dest_left_eye_points[:, 0].max(), dest_left_eye_points[:, 1].min()], 
                                    dest_left_eye_points.max(axis=0), [dest_left_eye_points[:, 0].min(), dest_left_eye_points[:, 1].max()]])
    right_eye_points = right_eye_contour.reshape(-1, 2).astype(np.float32)
    right_eye_points = np.float32([right_eye_points.min(axis=0), [right_eye_points[:, 0].max(), right_eye_points[:, 1].min()], 
                                right_eye_points.max(axis=0), [right_eye_points[:, 0].min(), right_eye_points[:, 1].max()]])
    dest_right_eye_points = dest_right_eye_contour.reshape(-1, 2).astype(np.float32)
    dest_right_eye_points = np.float32([dest_right_eye_points.min(axis=0), [dest_right_eye_points[:, 0].max(), dest_right_eye_points[:, 1].min()], 
                                        dest_right_eye_points.max(axis=0), [dest_right_eye_points[:, 0].min(), dest_right_eye_points[:, 1].max()]])
        
    # Perform perspective transform
    H_left, _ = cv2.findHomography(left_eye_points, dest_left_eye_points)
    warped_left_eye = cv2.warpPerspective(left_eye_region, H_left, (copy_dest_image.shape[1], copy_dest_image.shape[0]))

    H_right, _ = cv2.findHomography(right_eye_points, dest_right_eye_points)
    warped_right_eye = cv2.warpPerspective(right_eye_region, H_right, (copy_dest_image.shape[1], copy_dest_image.shape[0]))

    # Create masks for warped eye regions
    left_eye_warped_mask = np.zeros_like(copy_dest_image, dtype=np.uint8)
    cv2.drawContours(left_eye_warped_mask, [dest_left_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    right_eye_warped_mask = np.zeros_like(copy_dest_image, dtype=np.uint8)
    cv2.drawContours(right_eye_warped_mask, [dest_right_eye_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Combine images using masking
    combined_image = np.copy(copy_dest_image)
    cv2.copyTo(warped_left_eye, left_eye_warped_mask, combined_image)
    cv2.copyTo(warped_right_eye, right_eye_warped_mask, combined_image)
    
    cv2.imwrite('result_eye_overlay.jpg', combined_image)
    
    return combined_image
                                               
# if __name__ == "__main__":
#     extract()