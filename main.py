from lobe import ImageModel
import cv2
import os
from PIL import Image


model = ImageModel.load(os.getcwd())
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('image.jpg',frame)

image_path = os.path.join(os.getcwd(), "image.jpg")
img = Image.open(image_path)
result = model.predict(img)

if int(result.labels[0][1]*100) > 75:
    print(result.prediction)

else:
    print("other")
# if int(result["Labels"][0][result["Prediction"]]*100) > 75:
#     print(result.prediction)

# else:
#     print("other")

print(result)
for label, confidence in result.labels:
    print(f"{label}: {int(confidence*100)}%")


heatmap = model.visualize(img)
heatmap.show()