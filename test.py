'''
def generate_frames():
    if not camera.start():
        return

    frame_count = 0
    skip_frames = 2  # 每隔2帧进行一次推理
    last_annotated_frame = None

    while True:
        frame = camera.get_frame()
        if frame is not None:
            try:
                frame_count += 1
                if frame_count % skip_frames == 0:
                    # 执行YOLO推理
                    results = model(frame)
                    last_annotated_frame = results[0].plot()

                # 使用最新的推理结果或原始帧
                output_frame = last_annotated_frame #if last_annotated_frame is not None else frame
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"生成帧错误: {e}")
                time.sleep(0.01)
                '''

'''
这是pytorch mobilenetV3_large 的代码
def generate_frames():
    if not camera.start():
        return

    # 加载PyTorch情绪识别模型
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    emotion_model = torch.load('/path/to/your/model.pth', map_location=device)
    emotion_model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 情绪标签（按照训练时的顺序）
    emotion_labels = ['active', 'relaxed', 'anxious']
    emotion_colors = {
        'active': (0, 255, 0),  # 绿色
        'relaxed': (255, 165, 0),  # 橙色
        'anxious': (0, 0, 255)  # 红色
    }

    frame_count = 0
    skip_frames = 3  # 每三帧进行一次推理
    last_annotated_frame = None

    while True:
        frame = camera.get_frame()
        if frame is not None:
            try:
                frame_count += 1
                if frame_count % skip_frames == 0:  # 每三帧进行一次推理
                    # YOLO推理
                    results = model(frame)
                    annotated_frame = frame.copy()

                    # 处理每个检测到的目标
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if box.cls[0] == 0:  # 狗的类别索引
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])

                                # 裁剪狗的图像
                                dog_img = frame[y1:y2, x1:x2]

                                # PyTorch模型推理
                                with torch.no_grad():
                                    # 预处理图像
                                    img_tensor = transform(dog_img).unsqueeze(0).to(device)

                                    # 模型推理
                                    outputs = emotion_model(img_tensor)
                                    probabilities = torch.softmax(outputs, dim=1)
                                    emotion_idx = torch.argmax(probabilities[0]).item()
                                    emotion = emotion_labels[emotion_idx]
                                    emotion_conf = float(probabilities[0][emotion_idx])

                                # 绘制边界框
                                color = emotion_colors[emotion]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                                # 添加文本背景
                                label = f"Dog: {conf:.2f} | {emotion}: {emotion_conf:.2f}"
                                (label_w, label_h), _ = cv2.getTextSize(label,
                                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                                        0.6, 1)
                                cv2.rectangle(annotated_frame,
                                              (x1, y1 - 30),
                                              (x1 + label_w, y1),
                                              color, -1)

                                # 添加文本
                                cv2.putText(annotated_frame, label,
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 255, 255), 2)

                    last_annotated_frame = annotated_frame

                # 使用最新的推理结果或原始帧
                output_frame = last_annotated_frame if last_annotated_frame is not None else frame
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"生成帧错误: {e}")
                time.sleep(0.01)
                '''

'''
def preprocess_dog_image(dog_img):
    """
    对狗狗图像进行预处理，提高识别准确率
    """
    try:
        # 1. 确保图像尺寸合适
        if dog_img.shape[0] < 10 or dog_img.shape[1] < 10:  
            return None
            
        # 2. 亮度和对比度调整
        lab = cv2.cvtColor(dog_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. 降噪
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # 4. 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        
        return pil_img
    except Exception as e:
        print(f"图像预处理错误: {e}")
        return None

def generate_frames():
    if not camera.start():
        return

    # 加载PyTorch情绪识别模型
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    emotion_model = torch.load('/Users/mikeaalee/Desktop/dataset_test/model_epoch_23_acc_0.536.pth', map_location=device)
    emotion_model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 情绪标签
    emotion_labels = ['active', 'relaxed', 'anxious']
    emotion_colors = {
        'active': (0, 255, 0),    # 绿色
        'relaxed': (255, 165, 0), # 橙色
        'anxious': (0, 0, 255)    # 红色
    }

    frame_count = 0
    skip_frames = 3  # 每三帧进行一次推理
    last_annotated_frame = None

    while True:
        frame = camera.get_frame()
        if frame is not None:
            try:
                frame_count += 1
                if frame_count % skip_frames == 0:
                    # YOLO推理
                    results = model(frame)
                    annotated_frame = frame.copy()

                    # 处理每个检测到的目标
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if box.cls[0] == 0:  # 狗的类别索引
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])

                                # 裁剪狗的图像
                                dog_img = frame[y1:y2, x1:x2]
                                
                                # 预处理图像
                                processed_img = preprocess_dog_image(dog_img)
                                if processed_img is None:
                                    continue

                                # PyTorch模型推理
                                with torch.no_grad():
                                    # 多次采样预测
                                    predictions = []
                                    num_samples = 3
                                    
                                    for _ in range(num_samples):
                                        img_tensor = transform(processed_img).unsqueeze(0).to(device)
                                        outputs = emotion_model(img_tensor)
                                        probabilities = torch.softmax(outputs, dim=1)
                                        predictions.append(probabilities)
                                    
                                    # 平均多次预测结果
                                    avg_probabilities = torch.mean(torch.stack(predictions), dim=0)
                                    emotion_idx = torch.argmax(avg_probabilities[0]).item()
                                    emotion = emotion_labels[emotion_idx]
                                    emotion_conf = float(avg_probabilities[0][emotion_idx])

                                    # 只有当置信度超过阈值时才显示结果
                                    if emotion_conf > 0.6:
                                        # 绘制边界框
                                        color = emotion_colors[emotion]
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                                        # 添加文本背景
                                        label = f"Dog: {conf:.2f} | {emotion}: {emotion_conf:.2f}"
                                        (label_w, label_h), _ = cv2.getTextSize(label,
                                                                              cv2.FONT_HERSHEY_SIMPLEX,
                                                                              0.6, 1)
                                        cv2.rectangle(annotated_frame,
                                                    (x1, y1 - 30),
                                                    (x1 + label_w, y1),
                                                    color, -1)

                                        # 添加文本
                                        cv2.putText(annotated_frame, label,
                                                  (x1, y1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.6, (255, 255, 255), 2)

                    last_annotated_frame = annotated_frame
                else:
                    # 不进行推理的帧，使用原始帧
                    last_annotated_frame = frame

                # 使用最新的推理结果或原始帧
                output_frame = last_annotated_frame if last_annotated_frame is not None else frame
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"生成帧错误: {e}")
                time.sleep(0.01)
'''



