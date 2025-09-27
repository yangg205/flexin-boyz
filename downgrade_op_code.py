# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO('/home/jetbot/flexin-boyz/model/best.pt')

model.export(format='onnx', opset=12, imgsz=640)

import onnx

# --- CẤU HÌNH ---
# Đường dẫn tới file ONNX bị lỗi (file bạn vừa export từ YOLOv8)
INPUT_ONNX_FILE = "/home/jetbot/flexin-boyz/model/best.onnx"

# Đường dẫn để lưu file ONNX đã được sửa lỗi
# Đặt tên khác để không ghi đè lên file gốc, phòng trường hợp cần làm lại
OUTPUT_ONNX_FILE = "/home/jetbot/flexin-boyz/model/best_jetsons_compatible1.onnx"

# Phiên bản IR_VERSION mục tiêu.
# Jetson của bạn hỗ trợ tối đa là 8, nên chúng ta sẽ đặt là 8.
TARGET_IR_VERSION = 8
# --- KẾT THÚC CẤU HÌNH ---


def downgrade_ir_version(input_path, output_path, target_version):
    """
    Mở một file ONNX, hạ cấp IR_VERSION của nó, và lưu lại.
    """
    try:
        # 1. Tải mô hình ONNX
        print(f"Đang tải mô hình từ: {input_path}")
        model = onnx.load(input_path)

        # Lấy thông tin phiên bản hiện tại để so sánh
        original_ir_version = model.ir_version
        print(f"Phiên bản IR gốc: {original_ir_version}")

        if original_ir_version <= target_version:
            print("Phiên bản IR của mô hình đã thấp hơn hoặc bằng phiên bản mục tiêu. Không cần làm gì.")
            # Nếu muốn, bạn có thể sao chép file thay vì bỏ qua
            # import shutil
            # shutil.copy(input_path, output_path)
            return

        # 2. Hạ cấp phiên bản IR
        print(f"Hạ cấp IR version từ {original_ir_version} xuống {target_version}...")
        model.ir_version = target_version
        model.opset_import[0].version = 12

        # 3. Lưu lại mô hình đã được sửa đổi
        onnx.save(model, output_path)
        print(f"Đã lưu mô hình tương thích cho Jetson tại: {output_path}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    input_file_path = INPUT_ONNX_FILE
    downgrade_ir_version(input_file_path, OUTPUT_ONNX_FILE, TARGET_IR_VERSION)
