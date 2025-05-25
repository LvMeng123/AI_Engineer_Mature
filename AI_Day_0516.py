import cv2

# 打开默认摄像头（通常是0，外部摄像头可能是1或更高）
cap = cv2.VideoCapture(1)
print('cv.version', cv2.__version__)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头，请检查连接或权限")
    exit()

# 循环读取视频帧
while True:
    # 读取一帧
    ret, frame = cap.read()
    
    # 如果读取失败，退出
    if not ret:
        print("无法读取视频帧")
        break
    
    # 显示帧
    cv2.imshow('Camera Feed', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()