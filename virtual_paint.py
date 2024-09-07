import cv2
import os

# Initialize the folder path where header images are stored
folder_path = r"header"
header_lst = os.listdir(folder_path)

# Load overlay images from folder
overlay_lst = []
for img_path in header_lst:
    header_img = cv2.imread(os.path.join(folder_path, img_path))
    overlay_lst.append(header_img)


header = overlay_lst[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img[0:100, 0:1280] = header
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
