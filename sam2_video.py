import cv2, logging
from modules.eyes import Eyes 
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

def main():
    # Load the image
    image = cv2.imread('sam_test_image.jpg')

    # Call the function (assuming it's part of a class named `FaceMesh`)
    eyes_processor = Eyes() # Initialize the FaceMesh class
    img_h, img_w, _ = image.shape
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = eyes_processor.face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                
                # "left_eye": [7, 163, 144, 153, 154, 155, 173, 157, 158, 159, 160, 161],
                # "right_eye": [384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 398]
                if idx in [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                                         362, 398, 384, 385, 386, 387, 388, 390, 373, 374, 380, 466]:
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw small circle at each landmark point for eyes

                    # Save the output image
                    cv2.imwrite('results_sam_test_image.jpg', image)

if __name__ == "__main__":
    main()