from ultralytics import YOLO # Import thư viện YOLO từ ultralytics để nhận diện đối tượng

import cv2 # Import OpenCV để xử lý video và hình ảnh

import supervision as sv # Import supervision để hỗ trợ theo dõi và quản lý đối tượng

import numpy as np # Import NumPy để xử lý mảng và tính toán nhanh


class DetectionModel: # Định nghĩa lớp DetectionModel để xử lý nhận diện và theo dõi xe

    def __init__(self, model_path='models/yolov8n.pt'): # Hàm khởi tạo với đường dẫn tới mô hình YOLO

        self.model = YOLO(model_path) # Tải mô hình YOLO từ file đã chỉ định

        self.tracker = sv.ByteTrack() # Khởi tạo ByteTrack để theo dõi các đối tượng qua các khung hình

        self.cap = None # Biến để lưu đối tượng VideoCapture (chưa khởi tạo)

        self.counts = {'entering': 0, 'leaving': 0} # Từ điển đếm số xe vào (entering) và ra (leaving)

        self.speeds = {} # Từ điển lưu thông tin tốc độ và vị trí trước đó của từng xe

        self.line_y = None # Vị trí đường kẻ ngang để đếm xe (sẽ được tính sau)

        self.fps = None # Frames per second (số khung hình mỗi giây) của video

        self.vehicle_classes = [2, 3, 5, 7] # Danh sách các lớp xe từ COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
        
        
    def open_video(self, video_path): # Hàm mở file video để xử lý

        self.cap = cv2.VideoCapture(video_path) # Tạo đối tượng VideoCapture từ đường dẫn video

        if self.cap.isOpened(): # Kiểm tra xem video có mở thành công không

            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Lấy chiều cao của video

            self.line_y =  int(h * 0.7) # Đặt đường kẻ ngang ở giữa khung hình

            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) # Lấy FPS của video

            self.counts = {'entering': 0, 'leaving': 0} # Reset bộ đếm khi mở video mới

            self.speeds = {} # Reset thông tin tốc độ khi mở video mới

        return self.cap # Trả về đối tượng VideoCapture
    
    def detect_frame(self, frame): # Hàm xử lý từng khung hình để nhận diện và theo dõi xe

        scale_factor = 0.75 # Tăng hệ số thu nhỏ từ 0.5 lên 0.75 để nhận diện xe máy tốt hơn

        processing_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor) # Thu nhỏ khung hình


        # Nhận diện các đối tượng xe với ngưỡng confidence thấp hơn

        results = self.model(processing_frame, verbose=False, classes=self.vehicle_classes, conf=0.25)[0] # Giảm conf xuống 0.25

        detections = sv.Detections.from_ultralytics(results) # Chuyển kết quả YOLO sang định dạng supervision


        # Lọc các đối tượng có độ tin cậy > 0.25 để nhận diện xe máy tốt hơn

        mask = detections.confidence > 0.1 # Giảm ngưỡng từ 0.3 xuống 0.25 để phát hiện xe máy dễ hơn

        detections = detections[mask] # Áp dụng mask để lọc detections


        # Đảm bảo chỉ đếm các lớp xe (car, motorcycle, bus, truck)

        if detections.class_id is not None: # Kiểm tra xem có class_id không

            mask = np.isin(detections.class_id, self.vehicle_classes) # Lọc chỉ các lớp xe

            detections = detections[mask] # Áp dụng mask để giữ lại các xe


        tracked_detections = self.tracker.update_with_detections(detections) # Theo dõi các đối tượng qua ByteTrack


        # Phóng to lại tọa độ về kích thước gốc

        if tracked_detections.xyxy.size > 0: # Kiểm tra xem có đối tượng nào được theo dõi không

            tracked_detections.xyxy /= scale_factor # Phóng to tọa độ về kích thước khung hình gốc


        for i in range(len(tracked_detections)): # Lặp qua từng đối tượng được theo dõi

            track_id = tracked_detections.tracker_id[i] # Lấy ID của đối tượng

            bbox = tracked_detections.xyxy[i] # Lấy tọa độ hộp bao (bounding box)

            center_y = (bbox[1] + bbox[3]) / 2 # Tính tọa độ y trung tâm của hộp bao

            class_id = tracked_detections.class_id[i] # Lấy class ID để kiểm tra (car, motorcycle, etc.)


            if track_id not in self.speeds: # Nếu ID này chưa có trong từ điển tốc độ

                self.speeds[track_id] = {'prev_y': center_y, 'crossed': False} # Khởi tạo thông tin


            prev_y = self.speeds[track_id]['prev_y'] # Lấy vị trí y trước đó
            
            dy = center_y - prev_y

            if not self.speeds[track_id]['crossed']: # Nếu xe chưa vượt qua đường kẻ

                if dy > 0 and prev_y < self.line_y <= center_y: # Xe đi từ trên xuống (vào)

                    self.counts['entering'] += 1 # Tăng số xe vào

                    self.speeds[track_id]['crossed'] = True # Đánh dấu đã vượt qua

                elif dy < 0 and prev_y > self.line_y >= center_y: # Xe đi từ dưới lên (ra)

                    self.counts['leaving'] += 1 # Tăng số xe ra

                    self.speeds[track_id]['crossed'] = True # Đánh dấu đã vượt qua
                    
           

            
            
            # Tính tốc độ (km/h) dựa trên khoảng cách di chuyển và FPS

            speed = min(abs(center_y - prev_y) * self.fps / 3.6, 200)  # Giới hạn tốc độ tối đa 200 km/h

            self.speeds[track_id]['prev_y'] = center_y # Cập nhật vị trí y hiện tại

            self.speeds[track_id]['speed'] = speed # Lưu tốc độ


            # Vẽ hộp bao quanh xe (màu xanh lá)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),

            (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1) # Độ dày 1 để nhẹ hơn

            # Vẽ ID, tốc độ và class ID phía trên hộp bao để debug

            cv2.putText(frame, f"ID:{track_id} Cls:{class_id} {speed:.1f}km/h",

            (int(bbox[0]), int(bbox[1] - 10)),

            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Màu xanh lá, độ dày 1


        # Vẽ đường kẻ ngang (màu đỏ)

        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 0, 255), 1)

        # Vẽ số lượng xe vào/ra với màu trắng để rõ nét

        cv2.putText(frame, f"In:{self.counts['entering']} Out:{self.counts['leaving']}",

        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3) # Màu trắng, độ dày 3


        return frame # Trả về khung hình đã xử lý
    
    