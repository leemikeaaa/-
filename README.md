 ## 该项目是针对AX7z020的双目摄像头以太网传输的官方例程编写的接收端脚本
 ### 介绍各个文件用途
 1. udp_camera.py 主要是对udp数据包的接收并用flask框架在本地展示视频内容
 2. udp_ai.py 在udp_camera.py基础上利用ai工具加入了检错机制和一些差错纠错
 3. udp_yolo.py 在1的基础上加入yolo模型识别狗
 4. test.py 对视频帧的一些处理代码备用
 5. test2.py在3基础上加入了情绪识别模型分为四类,但是在测试集上表现一般准确度有50%左右而且在无独显的笔记本上运行效果较为卡顿
 6. test3.py进行测试yolo模型

#### 数据集链接链接: https://pan.baidu.com/s/1bCOj0xwLNA6hsJrSVJg16g?pwd=es2v 提取码: es2v 
https://github.com/leemikeaaa/-/issues/1#issue-2914109469
