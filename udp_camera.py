import socket
import random
import struct
import numpy as np
import cv2
from flask import Flask, Response, jsonify
from threading import Thread
import time

class UDPCamera:
    def __init__(self):
        # 基本配置
        self.local_port = 8080
        self.device_ip = "192.168.1.104"
        self.device_port = 8080
        self.mac_address = None
        self.device_ip_bytes = None
        
        # 图像参数
        self.width = 1280
        self.height = 720
        self.frame_size = self.width * self.height * 3  # RGB格式
        self.current_frame = None
        
        # 帧缓存优化
        self.frame_buffer = {}  # 使用字典存储数据包
        self.current_frame_id = 0
        self.packets_per_frame = (self.frame_size + 1431) // 1432  # 每帧的预期包数
        self.current_packets = 0
        
        # 帧率计算
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        
        # 创建UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65535 * 50)  # 增加缓冲区
        self.sock.bind(('0.0.0.0', self.local_port))
        
        # 启动接收线程
        self.running = True
        self.receive_thread = Thread(target=self.receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def generate_header(self):
        return random.randint(0, 127) << 1

    def send_inquiry(self):
        header = self.generate_header()
        command = struct.pack('!B I', header, 0x00020001)#对应udp的编码规则
        self.sock.sendto(command, (self.device_ip, self.device_port))
        return header

    def send_control(self, header):
        command = struct.pack('!B I 6s B B', 
                            header,
                            0x00020002,
                            self.mac_address,
                            0x01,
                            0x01)
        self.sock.sendto(command, (self.device_ip, self.device_port))

    def process_inquiry_response(self, data):
        header, cmd, mac, ip, end = struct.unpack('!B I 6s 4s B', data)
        self.mac_address = mac
        self.device_ip_bytes = ip
        return (header >> 1) == (self.last_header >> 1)

    def process_image_data(self, data):
        try:
            header = data[0]
            channel = data[4]
            seq_num = int.from_bytes(data[5:8], 'big')
            image_data = data[8:]
            
            # 计算帧ID和包ID
            frame_id = seq_num // self.packets_per_frame
            packet_id = seq_num % self.packets_per_frame
            
            # 如果是新帧
            if frame_id != self.current_frame_id:
                # 尝试处理当前帧
                self.try_process_frame()
                # 开始新帧
                self.current_frame_id = frame_id
                self.frame_buffer.clear()
                self.current_packets = 0
            
            # 存储数据包
            if packet_id not in self.frame_buffer:
                self.frame_buffer[packet_id] = image_data
                self.current_packets += 1
            
            # 如果收集到足够的包，尝试处理帧
            if self.current_packets >= self.packets_per_frame * 0.95:  # 允许5%的丢包
                self.try_process_frame()
                
        except Exception as e:
            print(f"处理图像数据错误: {e}")

    def try_process_frame(self):
        if not self.frame_buffer:
            return
            
        try:
            # 按顺序组合数据包
            frame_data = bytearray()
            sorted_packets = sorted(self.frame_buffer.items())
            for _, packet_data in sorted_packets:
                frame_data.extend(packet_data)
            
            # 确保数据长度足够
            if len(frame_data) >= self.frame_size:
                # 截取正确大小的数据
                frame_data = frame_data[:self.frame_size]
                
                # 转换为图像
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))
                
                # 计算帧率
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.fps_time > 1:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = current_time
                
                # 显示帧率和包信息
                cv2.putText(frame, f'FPS: {self.fps} Packets: {self.current_packets}/{self.packets_per_frame}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.current_frame = frame
                print(f"帧 {self.current_frame_id} 完成，包数: {self.current_packets}/{self.packets_per_frame}")
                
        except Exception as e:
            print(f"处理帧错误: {e}")
        
        # 清理缓存
        self.frame_buffer.clear()
        self.current_packets = 0

    def receive_data(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                if len(data) == 16:  # 询问应答
                    if self.process_inquiry_response(data):
                        self.send_control(self.last_header)
                else:  # 图像数据
                    self.process_image_data(data)
            except Exception as e:
                print(f"接收错误: {e}")

    def start(self):
        self.last_header = self.send_inquiry()

    def get_frame(self):
        return self.current_frame

    def get_fps(self):
        return self.fps

    def __del__(self):
        self.running = False
        self.sock.close()

# Flask应用
app = Flask(__name__)
camera = UDPCamera()

def generate_frames():
    camera.start()
    while True:
        frame = camera.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_fps')
def get_fps():
    return jsonify({'fps': camera.get_fps()})

@app.route('/')
def index():
    return """
    <html>
    <head>
        <style>
            .container {
                position: relative;
                display: inline-block;
            }
            .fps-display {
                position: absolute;
                top: 10px;
                left: 10px;
                background-color: rgba(0, 0, 0, 0.5);
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
        </style>
        <script>
            function updateFPS() {
                fetch('/get_fps')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('fps').innerText = 'FPS: ' + data.fps;
                    });
            }
            setInterval(updateFPS, 1000);
        </script>
    </head>
    <body>
        <h1>实时视频流</h1>
        <div class="container">
            <img src="/video_feed">
            <div id="fps" class="fps-display">FPS: 0</div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
