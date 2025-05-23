#读取文件中图片并显示
import cv2
import os

floder_path = "E:\wallpaper"
Image_type = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

if not os.path.isdir(floder_path):
    print('the floder is not exist')
else:
    filename = os.listdir(floder_path)[10]
    list_path = os.path.join(floder_path, filename)
    if os.path.isfile(list_path) and os.path.splitext(filename)[1].lower() in Image_type:
        img = cv2.imread(list_path,cv2.IMREAD_GRAYSCALE)
        cv2.imshow('filename', img)
        cv2.waitKey(0) & 0xff == ord('q')
        cv2.destroyAllWindows()
