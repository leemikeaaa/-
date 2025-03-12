from flask import Flask, render_template, Response, jsonify
import socket
import struct
import random
import time
import cv2
import numpy as np
import threading
import platform
import subprocess
import os
import traceback
from collections import defaultdict

app = Flask(__name__)


class UDPController:
    def __init__(self):
        # 配置参数
        self.BOARD_IP = '192.168.1.103'
        self.BOARD_PORT = 8080
        self.LOCAL_PORT = 8080
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480
        self.CHANNELS = 3
        self.MAX_RETRIES = 3
        self.BUFFER_SIZE = 65507  # 最大UDP数据包大小

        # 状态变量
        self.last_header = None
        self.board_mac = None
        self.running = True
        self.connected = False
        self.current_frames = {2: None, 3: None}

        # 创建UDP socket并增加接收缓冲区大小
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 设置socket缓冲区大小
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)  # 8MB接收缓冲区
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)  # 8MB发送缓冲区
            self.sock.bind(('0.0.0.0', self.LOCAL_PORT))
            self.sock.settimeout(2)

            # 验证缓冲区大小
            actual_recv_buf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            actual_send_buf = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            print(f"UDP socket绑定到端口 {self.LOCAL_PORT}")
            print(f"接收缓冲区大小: {actual_recv_buf} bytes")
            print(f"发送缓冲区大小: {actual_send_buf} bytes")
        except Exception as e:
            print(f"Socket初始化失败: {e}")
            raise

    def generate_header(self):
        """生成随机header"""
        random_bits = random.randint(0, 127)
        header = random_bits << 1
        self.last_header = header
        return header

    def check_network_connection(self):
        """检查网络连接"""
        try:
            print(f"正在检查网络连接 {self.BOARD_IP}:{self.BOARD_PORT}")
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_socket.settimeout(1)
            test_socket.sendto(b'test', (self.BOARD_IP, self.BOARD_PORT))
            test_socket.close()
            return True
        except Exception as e:
            print(f"网络检查错误: {e}")
            return False

    def parse_inquiry_response(self, response):
        """解析询问命令的应答"""
        try:
            if len(response) < 16:
                print(f"应答数据长度不足: {len(response)}字节")
                return False

            resp_header = response[0]
            fixed_value = struct.unpack('!L', response[1:5])[0]
            mac_bytes = response[5:11]
            self.board_mac = mac_bytes
            ip_bytes = response[11:15]
            end_mark = response[15]

            print("\n解析询问应答:")
            print(f"Header: 0x{resp_header:02X}")
            print(f"固定值: 0x{fixed_value:08X}")
            print(f"MAC地址: {':'.join([f'{b:02X}' for b in mac_bytes])}")
            print(f"IP地址: {'.'.join([str(b) for b in ip_bytes])}")
            print(f"结束标志: 0x{end_mark:02X}")

            return True

        except Exception as e:
            print(f"解析询问应答错误: {e}")
            print(f"应答数据: {response.hex()}")
            return False

    def send_inquiry(self):
        """发送询问命令"""
        for retry in range(self.MAX_RETRIES):
            try:
                if not self.check_network_connection():
                    print(f"网络连接失败 (尝试 {retry + 1}/{self.MAX_RETRIES})")
                    time.sleep(1)
                    continue

                header = self.generate_header()
                command = struct.pack('!BL', header, 0x00020001)

                print(f"\n发送询问命令 (尝试 {retry + 1}/{self.MAX_RETRIES})")
                print(f"Header=0x{header:02X}, Data=0x00020001")

                self.sock.sendto(command, (self.BOARD_IP, self.BOARD_PORT))

                try:
                    response, _ = self.sock.recvfrom(self.BUFFER_SIZE)
                    if self.parse_inquiry_response(response):
                        self.connected = True
                        return True
                except socket.timeout:
                    print("接收超时")

            except socket.error as e:
                print(f"网络错误 (尝试 {retry + 1}/{self.MAX_RETRIES}): {e}")
                time.sleep(1)

        print("发送询问命令失败，达到最大重试次数")
        return False

    def send_control(self):
        """发送控制命令"""
        for retry in range(self.MAX_RETRIES):
            try:
                if not self.board_mac:
                    print("错误: 未获取到板卡MAC地址")
                    return False

                command = struct.pack('!BL6sBB',
                                      self.last_header,
                                      0x00020002,
                                      self.board_mac,
                                      0x01,
                                      0x01)

                print(f"\n发送控制命令 (尝试 {retry + 1}/{self.MAX_RETRIES})")
                print(f"Header=0x{self.last_header:02X}")
                print(f"MAC地址: {':'.join([f'{b:02X}' for b in self.board_mac])}")

                self.sock.sendto(command, (self.BOARD_IP, self.BOARD_PORT))

                response, _ = self.sock.recvfrom(self.BUFFER_SIZE)
                print("收到控制命令应答")
                return True

            except socket.error as e:
                print(f"网络错误 (尝试 {retry + 1}/{self.MAX_RETRIES}): {e}")
                time.sleep(1)

        print("发送控制命令失败，达到最大重试次数")
        return False

    def parse_image_data(self, data):
        """解析图像数据包"""
        try:
            if len(data) < 8:
                print(f"数据包太短: {len(data)} bytes")
                return None

            header = data[0]
            fixed_bytes = data[1:4]
            channel_id = data[4]
            sequence = int.from_bytes(data[5:8], 'big')
            image_data = data[8:]

            # 打印详细的数据包信息
            print(f"数据包信息:")
            print(f"Header: 0x{header:02X}")
            print(f"Channel ID: {channel_id}")
            print(f"Sequence: {sequence}")
            print(f"数据长度: {len(image_data)}")

            if len(image_data) == 0:
                print("图像数据为空")
                return None

            # 验证channel_id
            if channel_id not in [2, 3]:
                print(f"无效的channel_id: {channel_id}")
                return None

            return {
                'header': header,
                'channel_id': channel_id,
                'sequence': sequence,
                'image_data': image_data
            }
        except Exception as e:
            print(f"解析图像数据错误: {str(e)}")
            print(f"数据包大小: {len(data)} bytes")
            print(f"数据内容: {data[:20].hex()}")
            return None

    def receive_images(self):
        """接收图像数据的线程函数"""
        print("\n开始接收图像数据...")
        self.sock.settimeout(0.1)

        # 使用defaultdict来自动创建新的序列号缓冲区
        image_buffers = {2: defaultdict(list), 3: defaultdict(list)}
        packet_counts = {2: defaultdict(int), 3: defaultdict(int)}
        last_packet_time = {2: 0, 3: 0}

        # 计算预期的图像大小和数据包数量
        image_size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT * 3
        expected_packets = (image_size + 1439) // 1440  # 向上取整

        print(f"预期图像大小: {image_size} bytes")
        print(f"预期数据包数量: {expected_packets}")

        while self.running:
            try:
                try:
                    data, addr = self.sock.recvfrom(self.BUFFER_SIZE)
                except socket.timeout:
                    # 检查超时的序列
                    current_time = time.time()
                    for channel in [2, 3]:
                        if current_time - last_packet_time[channel] > 1.0:
                            image_buffers[channel].clear()
                            packet_counts[channel].clear()
                    continue

                parsed_data = self.parse_image_data(data)
                if parsed_data is None:
                    continue

                channel_id = parsed_data['channel_id']
                sequence = parsed_data['sequence']
                image_data = parsed_data['image_data']

                # 更新最后接收时间
                last_packet_time[channel_id] = time.time()

                # 添加数据到缓冲区
                image_buffers[channel_id][sequence].append(image_data)
                packet_counts[channel_id][sequence] += 1

                current_packets = packet_counts[channel_id][sequence]
                print(f"Channel {channel_id}, Sequence {sequence}: "
                      f"Packet {current_packets}/{expected_packets}")

                # 检查是否收到完整的图像
                if current_packets >= expected_packets:
                    try:
                        # 合并数据包
                        complete_data = b''.join(image_buffers[channel_id][sequence])

                        # 确保数据大小正确
                        if len(complete_data) >= image_size:
                            # 截取正确大小的数据
                            complete_data = complete_data[:image_size]

                            # 转换为图像
                            frame = np.frombuffer(complete_data, dtype=np.uint8)
                            frame = frame.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3))

                            # RGB转BGR
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                            # 添加信息显示
                            cv2.putText(frame, f"Seq: {sequence}",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            self.current_frames[channel_id] = frame
                            print(f"成功更新Channel {channel_id}的帧")

                            # 清理缓冲区
                            del image_buffers[channel_id][sequence]
                            del packet_counts[channel_id][sequence]

                            # 清理旧的序列号
                            current_sequences = sorted(image_buffers[channel_id].keys())
                            if len(current_sequences) > 2:
                                for old_seq in current_sequences[:-2]:
                                    del image_buffers[channel_id][old_seq]
                                    del packet_counts[channel_id][old_seq]

                    except Exception as e:
                        print(f"处理图像错误: {str(e)}")
                        del image_buffers[channel_id][sequence]
                        del packet_counts[channel_id][sequence]

            except Exception as e:
                print(f"接收错误: {str(e)}")
                print(f"错误类型: {type(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                time.sleep(0.1)

    def start_receiving(self):
        """启动图像接收"""
        if not self.connected:
            if not self.send_inquiry() or not self.send_control():
                print("无法建立连接")
                return False

        self.receive_thread = threading.Thread(target=self.receive_images)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        print("开始接收图像数据")
        return True

    def get_frame(self, channel_id):
        """获取指定通道的当前帧"""
        frame = self.current_frames.get(channel_id)
        if frame is not None:
            try:
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    return jpeg.tobytes()
            except Exception as e:
                print(f"编码帧错误: {str(e)}")
        return None

    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        self.running = False
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join(timeout=1.0)
        self.sock.close()
        print("资源清理完成")


# Flask路由和控制器实例
controller = None


def init_controller():
    global controller
    try:
        controller = UDPController()
        if controller.start_receiving():
            print("UDP控制器初始化成功")
            return True
        else:
            print("UDP控制器初始化失败")
            return False
    except Exception as e:
        print(f"UDP控制器初始化错误: {e}")
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    global controller
    if controller is None:
        return jsonify({'status': 'not_initialized'})
    return jsonify({
        'status': 'connected' if controller.connected else 'disconnected',
        'ip': controller.BOARD_IP,
        'port': controller.BOARD_PORT
    })


@app.route('/reconnect')
def reconnect():
    global controller
    if controller is not None:
        controller.cleanup()
    success = init_controller()
    return jsonify({'success': success})


def gen_frame(channel_id):
    global controller
    while controller and controller.running:
        frame = controller.get_frame(channel_id)
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


@app.route('/video_feed/<int:channel_id>')
def video_feed(channel_id):
    return Response(gen_frame(channel_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        if init_controller():
            app.run(host='0.0.0.0', port=8000, threaded=True)
        else:
            print("程序初始化失败")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        if controller:
            controller.cleanup()