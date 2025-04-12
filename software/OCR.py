import pytesseract
from PIL import Image
import cv2
import pyttsx3

img = cv2.imread('D:\TTS\Bien-bao-chi-dan-giao-thong.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sharp_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
sharp_img = cv2.addWeighted(gray_img, 1.5, sharp_img, -0.5, 0)

# CẤU HÌNH đường dẫn tới tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\OCR\tesseract.exe'

# ĐỌC ảnh

text = pytesseract.image_to_string(sharp_img)
print(text)

# Chuyển văn bản thành giọng nói
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
engine.setProperty('rate', 150)  # Tốc độ nói
engine.setProperty('volume', 1)  # Âm lượng (0.0 đến 1.0)
engine.setProperty('voice', voices[1].id)  # Chọn giọng nữ (nếu có)
