from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model('dog2.jpg')

for r in results:
    boxes = r.boxes
    img = r.orig_img.copy()

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = r.names.get(class_id, "unknown")
        confidence = float(box.conf[0])

        if class_name == "dog":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 绘制矩形
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 调整文字位置、大小和颜色
            label = f"{class_name} {confidence:.2f}"
            y_text = max(y2 - 10, 10)  # 确保不超出图像顶部
            cv2.putText(img, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 所有框绘制完毕后显示图像
    cv2.imshow("Detected Dog", img)
    cv2.imwrite('detected2.jpg', img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()