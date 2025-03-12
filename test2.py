
import socket
import random
import struct
import numpy as np
import cv2
from flask import Flask, Response
from threading import Thread
import time
from tensorflow.keras.models import load_model

class UDPCamera:
    def __init__(self):
        # 网络参数
        self.local_port = 8080
        self.device_ip = "192.168.1.100"
        self.device_port = 8080
        self.mac_address = None
        self.device_ip_bytes = None
        self.last_header = None

        # 图像参数
        self.width = 1280
        self.height = 720
        self.frame_size = self.width * self.height * 3  # RGB格式
        self.packet_size = 1440
        self.current_frame = None
        self.frame_buffer = bytearray()

        # 统计信息
        self.frame_count = 0
        self.lost_packets = 0
        self.last_seq = -1

        # 创建UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.local_port))

        # 启动接收线程
        self.running = True
        self.receive_thread = Thread(target=self.receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def generate_header(self):
        # 生成随机header，保持bit0为0
        return random.randint(0, 127) << 1

    def send_inquiry(self):
        # 发送询问命令
        header = self.generate_header()
        command = struct.pack('!B I', header, 0x00020001)
        try:
            self.sock.sendto(command, (self.device_ip, self.device_port))
            print(f"发送询问命令: header={header:08b}")
            return header
        except Exception as e:
            print(f"发送询问命令失败: {e}")
            return None

    def send_control(self, header):
        # 发送控制命令
        try:
            command = struct.pack('!B I 6s B B',
                                  header,
                                  0x00020002,
                                  self.mac_address,
                                  0x01,
                                  0x01)
            self.sock.sendto(command, (self.device_ip, self.device_port))
            print("发送控制命令成功")
        except Exception as e:
            print(f"发送控制命令失败: {e}")

    def process_inquiry_response(self, data):
        try:
            # 解析询问应答
            header, cmd, mac, ip, end = struct.unpack('!B I 6s 4s B', data)
            self.mac_address = mac
            self.device_ip_bytes = ip
            print(f"收到询问应答: MAC={mac.hex()}, IP={socket.inet_ntoa(ip)}")
            return (header >> 1) == (self.last_header >> 1)
        except Exception as e:
            print(f"处理询问应答失败: {e}")
            return False

    def process_image_data(self, data):
        # 解析图像数据包
        if len(data) < 8:
            return

        try:
            header = data[0]
            channel = data[4]  # 2为摄像头1，3为摄像头2
            seq_num = int.from_bytes(data[5:8], 'big')
            image_data = data[8:]

            # 检测丢包
            if self.last_seq != 0 and seq_num != 1 and seq_num != (self.last_seq + 1):
                self.lost_packets += 1
                #print(f"检测到丢包: 上一个序号 {self.last_seq}, 当前序号 {seq_num}")

            self.last_seq = seq_num

            # 添加到帧缓冲区
            if seq_num == 1:
                self.frame_buffer = bytearray()
            self.frame_buffer.extend(image_data)

            # 检查是否接收到完整帧,将udp包封装成帧
            if len(self.frame_buffer) >= self.frame_size or seq_num==1920:
                frame = np.frombuffer(self.frame_buffer[:self.frame_size], dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))#变为rgb矩阵
                self.current_frame = frame
                self.frame_count += 1

                if self.frame_count % 30 == 0:
                 #print(f"已接收 {self.frame_count} 帧, 丢包数: {self.lost_packets}")
                    pass
                self.frame_buffer = self.frame_buffer[self.frame_size:]

        except Exception as e:
            #print(f"处理图像数据失败: {e}")
            pass
    def receive_data(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                if len(data) == 16:  # 询问应答
                    if self.process_inquiry_response(data):
                        self.send_control(self.last_header)
                elif len(data) >= 8:  # 图像数据
                    self.process_image_data(data)
            except Exception as e:
                print(f"接收数据错误: {e}")
                time.sleep(0.1)

    def start(self):
        self.last_header = self.send_inquiry()
        if self.last_header is None:
            print("启动失败：无法发送询问命令")
            return False
        return True

    def get_frame(self):
        return self.current_frame

    def __del__(self):
        self.running = False
        self.sock.close()

from ultralytics import YOLO
# Flask应用
app = Flask(__name__)
camera = UDPCamera()
model = YOLO(model="best.pt")
#yolo模型推理调用



'''
def generate_frames():
    if not camera.start():
        return

    while True:
        frame = camera.get_frame()
        if frame is not None:
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                #buffer=yolo_infer(buffer)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"生成帧错误: {e}")
                time.sleep(0.01)
'''


def generate_frames():
    if not camera.start():
        return

    # 加载情绪识别模型
    emotion_model = load_model('/Users/mikeaalee/Desktop/dataset/64.h5')
    emotion_labels = ['angry', 'happy', 'relaxed', 'sad']

    # 为不同情绪设置不同的颜色
    emotion_colors = {
        'angry': (0, 0, 255),  # 红色
        'happy': (0, 255, 0),  # 绿色
        'relaxed': (255, 165, 0),  # 橙色
        'sad': (255, 0, 0)  # 蓝色
    }

    frame_count = 0
    skip_frames = 2
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
                            # 检查是否为狗
                            if box.cls[0] == 0:  # 确认这是你模型中狗的类别索引
                                # 获取边界框坐标
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])

                                # 裁剪狗的图像
                                dog_img = frame[y1:y2, x1:x2]

                                # 处理图像用于情绪识别
                                processed_img = cv2.resize(dog_img, (224, 224))
                                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                processed_img = processed_img / 255.0
                                processed_img = np.expand_dims(processed_img, axis=0)

                                # 情绪预测
                                emotion_pred = emotion_model.predict(processed_img)
                                emotion_idx = np.argmax(emotion_pred[0])
                                emotion = emotion_labels[emotion_idx]
                                emotion_conf = float(emotion_pred[0][emotion_idx])

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

                # 使用最新的推理结果
                output_frame = last_annotated_frame if last_annotated_frame is not None else frame
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"生成帧错误: {e}")
                time.sleep(0.01)



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>OV5642摄像头视频流</title>
        <style>
            body { 
                text-align: center; 
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            h1 { color: #333; }
            img { 
                max-width: 100%;
                border: 2px solid #666;
                border-radius: 8px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>OV5642摄像头实时视频流</h1>
        <img src="/video_feed">
    </body>
    </html>
    """


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    except Exception as e:
        print(f"程序启动失败: {e}")
