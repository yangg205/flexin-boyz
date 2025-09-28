# Giải pháp của problema.py
##  Mục tiêu

    Xây dựng bộ điều khiển JetBot event-driven có khả năng:
    - Lấy bản đồ động từ API server bằng token và loại bản đồ (map_a, map_b, map_z).
    - Tự động lập kế hoạch đường đi từ điểm xuất phát >>> đích bằng giải thuật A*.
    - Bám line (line following) kết hợp xử lý tình huống ở giao lộ (intersection handling).
    - Sử dụng LiDAR để phát hiện sự kiện (ví dụ: giao lộ, chướng ngại vật).
    - Tự động ghi lại hành trình (video debug).

## Kiến trúc
    Trạng thái robot (RobotState)
    - Robot hoạt động theo các trạng thái rời rạc:
    - WAITING_FOR_LINE >>> chờ thấy vạch kẻ đường.
    - DRIVING_STRAIGHT >>> bám line đi thẳng.
    - APPROACHING_INTERSECTION >>> tiến vào giao lộ.
    - HANDLING_EVENT >>> xử lý quyết định rẽ/tránh/ngõ cụt.
    - LEAVING_INTERSECTION >>> rời giao lộ, đi thẳng thêm một đoạn.
    - REACQUIRING_LINE >>> tìm lại line sau khi rẽ.
    - DEAD_END >>> ngõ cụt hoặc lỗi.
    - GOAL_REACHED >>> đã đến đích.

### Khởi tạo
    Nhận tham số từ CLI:   --token <TEAM_TOKEN> --map_type <map_a|map_b|map_z>
    Gọi API https://hackathon2025-dev.fpt.edu.vn/api/maps/get_active_map/ để tải bản đồ.
    Dùng MapNavigator để phân tích graph và chạy A* tìm đường.
    Xác định start_node, end_node, planned_path.
    Nhận dữ liệu cảm biến

### Camera CSI (/csi_cam_0/image_raw):
    Chuyển đổi ảnh sang OpenCV.
    Tìm vạch line bằng HSV filter + contour trong ROI.

### LiDAR (/scan):
    Dùng SimpleOppositeDetector để phát hiện giao lộ/sự kiện.
    Logic di chuyển

### Bám line:
    Dùng PID-like correction để điều chỉnh động cơ theo vị trí vạch line trong ROI.

### Giao lộ:
    Khi LiDAR báo giao lộ: >>> Robot dừng.
    Cập nhật vị trí hiện tại (current_node_id).
    Nếu đã đến đích >>> chuyển sang GOAL_REACHED.
    Ngược lại >>> gọi handle_intersection().
    Xử lý rẽ tại giao lộ (handle_intersection): >>> Dùng bản đồ và đường đi đã lên kế hoạch (planned_path).
    Xác định hướng kế tiếp (N/E/S/W).
    Ánh xạ hướng tuyệt đối >>> hành động tương đối (thẳng/trái/phải).
    Điều khiển robot quay bằng turn_robot().
    Cập nhật hướng mới và node mục tiêu.
    Quay video debug
    Ghi lại khung hình camera kèm overlay:
    ROI (khu vực quét line).
    Trạng thái hiện tại.
    Đường line và trọng tâm phát hiện được.
    Xuất file jetbot_run.avi.

## Điểm nổi bật
    - Tích hợp bản đồ động từ server >>> cho phép cập nhật đường đi theo môi trường thực tế.
    - Kết hợp camera + LiDAR để đảm bảo phát hiện chính xác giao lộ.
    - Xử lý sự kiện theo trạng thái >>> giúp robot an toàn, dễ debug, tránh rẽ sai.
    - Tự động tái lập kế hoạch khi lỗi (ví dụ: không tìm thấy đường đi thì dừng tại DEAD_END).
    - Hỗ trợ Mock Robot để có thể test code mà không cần phần cứng.

## Luồng hoạt động tóm tắt
    1. Khởi tạo >>> lấy map từ server >>> chạy A* >>> lên kế hoạch.
    2. Đợi camera tìm thấy line >>> chuyển sang DRIVING_STRAIGHT.
    3. Bám line liên tục, điều chỉnh hướng.
    4. Khi LiDAR phát hiện giao lộ:
        - Dừng.
        - Cập nhật node.
        - Nếu chưa đến đích >>> tính toán hướng rẽ >>> quay robot.
    5. Rời giao lộ >>> tìm lại line >>> tiếp tục bám line.
    6. Khi đến end_node >>> dừng >>> kết thúc.





# Giải pháp của problemb.py

## Mục tiêu

    Xây dựng bộ điều khiển JetBot event-driven với khả năng:
        - Đọc bản đồ cục bộ (map.json) thay vì từ server.
        - Bám line bằng camera CSI.
        - Phát hiện giao lộ bằng LiDAR.
        - Xử lý biển báo (prescriptive/prohibitive) để điều chỉnh đường đi theo luật giao thông giả lập.
        - Xử lý dữ liệu (QR code, toán học) và gửi thông tin lên MQTT broker.
        - Tự động cập nhật đường đi nếu kế hoạch ban đầu bị cấm.

## Kiến trúc
 
### 1. Trạng thái robot (RobotState)
    Giống như bản A (problema.py):
    Chờ line → bám line → xử lý giao lộ → rời giao lộ → tìm lại line → đích/ngõ cụt.

### 2. Khởi tạo
    Đọc cấu hình trong setup_parameters():
        - Tốc độ, góc quay, ROI camera.
        - Thông tin MQTT (localhost:1883, topic jetbot/corrected_event_data).
        - Đường dẫn bản đồ: map.json.
        - Khởi tạo ROS node + đăng ký topic camera (/csi_cam_0/image_raw) và LiDAR (/scan).
        - Tạo MapNavigator từ map.json, tìm đường đi ban đầu (find_path).

### 3. Nhận dữ liệu cảm biến
    Camera CSI:
        Dùng HSV filter để phát hiện vạch line trong ROI.
        Áp dụng thêm focus mask để chỉ quét vùng trung tâm → giảm nhiễu.
    LiDAR:
        Dùng SimpleOppositeDetector để phát hiện giao lộ.

### 4. Logic điều khiển
    Line following:
        Nếu còn thấy vạch → điều chỉnh theo sai số (giới hạn bởi MAX_CORRECTION_ADJ).
        Giao lộ (intersection handling):
        Robot dừng → quét biển báo bằng camera.

    Xử lý dữ liệu (Step 2):
        Nếu phát hiện QR code → publish dữ liệu qua MQTT.
        Nếu phát hiện math problem → giải toán, publish kết quả.

    Điều hướng (Step 3):
        Lấy hướng từ bản đồ (planned_action).

    Kiểm tra biển báo bắt buộc (prescriptive):
        Ví dụ: L → buộc rẽ trái, R → buộc rẽ phải.
        Nếu khác với plan → đánh dấu là chệch hướng.

    Kiểm tra biển báo cấm (prohibitive):
        Nếu trùng với hành động dự định → hủy bỏ cạnh đó, cập nhật banned_edges và chạy lại A*.
        Nếu xung đột giữa bắt buộc và cấm → báo lỗi bản đồ, chuyển sang DEAD_END.

    Thực thi quyết định:
        Rẽ trái/phải/thẳng bằng turn_robot().
        Nếu đi chệch plan, xác định lại node mới → tái lập đường đi từ node đó.

### 5. MQTT Integration

    Khi gặp QR hoặc toán học → publish JSON data lên topic:

    { "type": "QR_CODE", "value": "simulated_data_123" }
    { "type": "MATH_PROBLEM", "value": "2+2=4" }
        
    Giúp backend lưu lại dữ liệu thu thập được dọc hành trình.

### 6. Video Debug

    Ghi video với overlay ROI + trạng thái robot (jetbot_run.avi).
    Ghi khung hình ngay cả khi robot đang quay tại giao lộ (turn_robot).

## Điểm nổi bật

    - Tích hợp biển báo & luật giao thông giả lập:
        Prescriptive (L, R, F) → ép hành động.
        Prohibitive (NL, NR, NF) → cấm hành động.
    - Khả năng tái lập kế hoạch động khi gặp biển báo cấm.
    - Xử lý tình huống chệch hướng khi buộc phải đi khác với A*.
    - Tích hợp MQTT để truyền dữ liệu thu thập (QR, toán học).
    - Hệ thống state machine rõ ràng giúp debug dễ dàng.

## Luồng hoạt động tóm tắt
    1. Khởi động → load map.json → tìm đường → vào WAITING_FOR_LINE.
    2. Camera phát hiện line → chuyển sang DRIVING_STRAIGHT.
    3. LiDAR phát hiện giao lộ → dừng lại.
    4. Quét camera tìm biển báo/QR:
        - Nếu QR → publish MQTT.
        - Nếu math → giải toán, publish MQTT.
        - Xác định hướng đi:
    5. Theo biển báo bắt buộc (nếu có).
    6. Nếu bị cấm → bỏ cạnh, tái lập kế hoạch.
    7. Nếu chệch hướng → cập nhật node, tái lập kế hoạch từ node mới.
    8. Thực hiện rẽ/đi thẳng.
    9. Lặp lại cho đến khi end_node → GOAL_REACHED.



# Giải pháp của problemc.py

## Mục tiêu
    Xây dựng bộ điều khiển JetBot event-driven với phiên bản rút gọn, tập trung vào:
        - Bám line cơ bản bằng camera CSI.
        - Phát hiện giao lộ bằng LiDAR.
        - Nhận diện QR code từ camera.
        - Publish dữ liệu qua MQTT.

## Kiến trúc
### 1. Trạng thái robot (RobotState)
    Giống các bản A/B, nhưng được sử dụng đơn giản hơn:
        WAITING_FOR_LINE — chờ thấy line.
        DRIVING_STRAIGHT — bám line.
        HANDLING_EVENT — khi gặp giao lộ.
        LEAVING_INTERSECTION — rời giao lộ.
        DEAD_END / GOAL_REACHED — kết thúc hành trình.

### 2. Khởi tạo
    Cấu hình trong setup_parameters():
        - Tham số bám line (ROI, ngưỡng màu line, tốc độ).
        - MQTT (localhost:1883, topic jetbot/corrected_event_data).
        - Định nghĩa tập prescriptive signs (L, R, F), prohibitive signs (NL, NR, NF), và data items (QR, math).
    ROS node:
        - Đăng ký camera (/csi_cam_0/image_raw) và LiDAR (/scan).
    Khởi tạo MQTT client.
    Không dùng bản đồ (MapNavigator) → robot không có lộ trình toàn cục.

### 3. Nhận dữ liệu cảm biến
    Camera CSI:
        - Chuyển ảnh sang HSV → lọc vạch line.
        - Giới hạn ROI ở vùng trung tâm để tăng ổn định.
    LiDAR:
        - SimpleOppositeDetector để phát hiện giao lộ.

### 4. Logic điều khiển
    Line following:
        - Nếu thấy vạch line trong ROI → tính toán sai số → điều chỉnh động cơ.
        - Nếu mất vạch → robot sẽ tạm thời giữ tốc độ thấp, hoặc dừng để tránh lệch hướng.
    Giao lộ (event handling):
        - Khi LiDAR báo giao lộ → robot dừng.
        - Gọi scan_for_signs():
            Nhận diện QR code bằng thư viện pyzbar.
            Ghi log các class sign đã train (qua YOLO_CLASS_MAPPING), nhưng chưa thực sự chạy YOLO.
            Nếu phát hiện QR → log & publish dữ liệu qua MQTT.
        - Quyết định xử lý giao lộ hiện tại rất đơn giản:
            Robot không chạy logic A* hoặc tái lập kế hoạch.
            Mặc định chuyển sang LEAVING_INTERSECTION.
    MQTT publish:
        - Khi thấy QR code → publish JSON data:
        { "class_name": "qr_code", "value": "<qr_data>" }

## 5. Video Debug
    Có hỗ trợ VideoWriter, nhưng ít overlay hơn so với bản A/B.
    Tập trung ghi lại hành trình cơ bản.

## Điểm nổi bật
    - Tối giản, không phụ thuộc vào bản đồ hay logic A*.
    - Tích hợp QR code scanning và gửi dữ liệu qua MQTT.
    - Vẫn giữ kiến trúc state machine, giúp dễ nâng cấp lên bản B hoặc A.
    - Thích hợp cho demo nhanh, training, hoặc chạy thử trong môi trường hẹp.

## Luồng hoạt động tóm tắt
    1. Khởi động → ROS node + MQTT.
    2. Chờ camera thấy line → vào DRIVING_STRAIGHT.
    3. Bám line liên tục.
    4. Khi LiDAR báo giao lộ:
        - Dừng.
        - Quét camera tìm QR code.
        - Nếu thấy → publish dữ liệu qua MQTT.
        - Sau đó → chuyển sang LEAVING_INTERSECTION.
    5. Tiếp tục bám line cho đến khi kết thúc (manual stop hoặc hết line).