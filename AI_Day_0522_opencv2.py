import cv2
import os

def load_file():
    path = 'E:/wallpaper'
    #加载文件、获取内容名称
    filename = os.listdir(path)[0]
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

#读取文件并保存
def opencv_imwrite():
    path = 'E:/wallpaper'
    #加载文件、获取内容名称
    filename = os.listdir(path)[0]
    image_path = os.path.join(path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test',image)
    cv2.waitKey(0) & 0XFF == ord('q')
    cv2.destroyAllWindows()
    cv2.imwrite('E:/wallpaper/save_image.png', image)

#显示图像到窗口并命名窗口
def opencv_imshow():
    file_path = 'E:/wallpaper'
    filename = os.listdir(file_path)[1]
    path = os.path.join(file_path, filename)
    image = cv2.imread(path)
    cv2.imshow('我的图像', image)
    cv2.waitKey(0) & 0XFF == ord('Q')
    cv2.destroyAllWindows()
    
#转换颜色空间
def opencv_ctvColor():
    path = 'E:/wallpaper'
    filename = os.listdir(path)[8]
    image_path = os.path.join(path, filename)
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#调整图像大小,最后的参数是计算这些新像素值的算法
def opencv_resize():
    path = 'E:/wallpaper'
    filename = os.listdir(path)[8]
    image_path = os.path.join(path, filename)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('缩小', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#翻转图像
def opencv_flip():
    image = cv2.resize(load_file(), None, fx=0.5, fy = 0.2, interpolation=cv2.INTER_LINEAR)
    flip_image = cv2.flip(image,0) #水平翻转
    cv2.imshow('flip_image',flip_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#图像旋转  
def opencv_rotate():
    img = load_file()
    image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resize_image = cv2.resize(image, None, fy=0.3, fx=0.3, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('resize_image', resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#图像旋转矩阵
def opencv_getrotationMatrix2D():
    image = load_file()
    h,w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2),90,0.2)
    rotated = cv2.warpAffine(image, M, (w,h))
    img = cv2.resize(rotated, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR_EXACT)
    cv2.imshow('旋转', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#二值化图像（设置一个阈值。遍历图像中的每一个像素）
def opencv_threshold():
    init_image = load_file()
    _, thresh = cv2.threshold(init_image, 123, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', cv2.resize(thresh, None, fx=0.5, fy=0.2, interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#高斯模糊降噪(去除图像中的高斯噪声，边缘会变得模糊)
#参数说明：第二个我参数是高斯核，只能是正奇数，最后一个参数是sigma选择，参数越大越模糊，高斯核和sigma两者谁为0，另一方都会自动计算
def opencv_GaussianBlur():
    init_image = load_file()
    image = cv2.GaussianBlur(init_image, (9,9), 0)
    cv2.imshow('mohu', cv2.resize(image, (2560, 1440), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#混合两张图像
def opencv_addWeighted():
    path = 'E:/wallpaper'
    #加载文件、获取内容名称
    filename = os.listdir(path)[1]
    filename1 = os.listdir(path)[2]
    image_path = os.path.join(path, filename)
    image_path2 = os.path.join(path,filename1)
    image = cv2.imread(image_path)
    image1 = cv2.imread(image_path2)
    image_final = cv2.addWeighted(image, 0.3, image1, 0.7, 0)
    cv2.imshow('hunhe',cv2.resize(image_final, (1960,1080), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#缩放并转换为8位图像，增强图像对比度（alpha缩放因子，用来调整对比度，>1对比度增加）
def opencv_convertScaleAbs():
    init_image = load_file()
    init_image = cv2.convertScaleAbs(init_image, alpha=1.5, beta=50)
    cv2.imshow('image', cv2.resize(init_image, (1920,1080), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
opencv_convertScaleAbs()