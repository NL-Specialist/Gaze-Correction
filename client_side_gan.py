import requests

response = requests.post("http://192.168.0.58:8021/load_model/")
print(response.json())

image_path = "/home/chatbot/GAN/Gaze Correction FYP/Duncan/away/validate/image_244/left_eye.jpg"
with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post("http://192.168.0.58:8021/generate_image/", files=files)
    
if response.status_code == 200:
    with open("CLIENT_generated_image.jpg", "wb") as f:
        f.write(response.content)
