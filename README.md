## JetBot Event-Driven Controllers

Repository này chứa ba biến thể bộ điều khiển **JetBot** được phát triển cho các kịch bản khác nhau của cuộc thi/hackathon.  
Các bộ điều khiển đều dựa trên kiến trúc **event-driven**, sử dụng ROS để giao tiếp với cảm biến (LiDAR, camera CSI), điều khiển động cơ, và hỗ trợ các tính năng mở rộng như lấy bản đồ từ server, nhận diện QR code / biển báo, và gửi dữ liệu qua MQTT.

---
## Yêu cầu môi trường

- Python 3.6+
- ROS (Robot Operating System)
- Các thư viện Python:
  - `rospy`
  - `opencv-python`
  - `numpy`
  - `requests`
  - `paho-mqtt`
  - `pyzbar`
- Phần cứng:
  - NVIDIA JetBot (hoặc Jetson Nano + driver robot)
  - Camera CSI
  - Cảm biến LiDAR
- Các module nội bộ (cần có trong cùng workspace):
  - `jetbot.Robot`
  - `opposite_detector.SimpleOppositeDetector`
  - `map_navigator.MapNavigator`
---

## Hướng dẫn chạy nhanh

Hãy kết nối vào jetbot, sau đó thực hiện lần lượt 3 câu lệnh sau trong terminal:

```sh
roslaunch jetbot_pro lidar.launch
roslaunch jetbot_pro csi_camera.launch
python3 ros_lidar_follower.py
```
cách chạy các problem 
python3 problema/b/c.py --token YOUR_TEAM_TOKEN --map map_z
(giả lập môi trường giả python --version 3.8)
source ~/yolo_env/bin/activate
---
## Chức năng
problema.py
- Điều kiển robot tự tìm đường từ điểm bắt đầu đến điểm kết thúc.
- Robot tự căn chỉnh để đi trên đường line.

ptoblemb.py
- Điểu kiển robot từ start qua các đỉnh "Load" 
- Đọc thông tin trên biển hiệu ở đỉnh "Load" và gửi về cho server.
- Lặp lại cho đến khi hết đỉnh và di chuyển đến điểm kết thúc.

problemc.py
- Điều kiển robot tự tìm đường từ điểm bắt đầu đến điểm kết thúc.
- Đọc biển báo và gửi về server, thực hiện di chuyển theo yêu cầu của biển báo.

---

## Các trạng thái robot

Tất cả các phiên bản đều chia trạng thái hoạt động theo RobotState:

WAITING_FOR_LINE — Chờ tìm thấy line.
DRIVING_STRAIGHT — Bám line đi thẳng.
APPROACHING_INTERSECTION — Tiến vào giao lộ.
HANDLING_EVENT — Xử lý giao lộ (biển báo, dữ liệu, hoặc chọn đường).
LEAVING_INTERSECTION — Rời khỏi giao lộ.
REACQUIRING_LINE — Tìm lại line sau giao lộ.
DEAD_END — Ngõ cụt hoặc lỗi.
GOAL_REACHED — Đã đến đích.
---