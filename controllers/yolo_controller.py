from flask import Flask, render_template, Response, request # Import Flask và các module liên quan

from werkzeug.utils import secure_filename # Import hàm để bảo mật tên file

from models.yolo_model import DetectionModel # Import lớp DetectionModel từ yolo_model.py

import cv2 # Import OpenCV để xử lý video

import os # Import os để quản lý file

from forms.form_opencv import UploadForm # Import form upload từ file form_opencv.py

from config import Config # Import cấu hình từ file config.py

detector = DetectionModel()


def generate_frames(video_path): # Hàm tạo luồng khung hình để stream video

    cap = detector.open_video(video_path) # Mở video bằng detector

    if not cap.isOpened(): # Kiểm tra nếu không mở được video

        return # Thoát hàm nếu lỗi


    while True: # Lặp qua từng khung hình

        ret, frame = cap.read() # Đọc khung hình từ video

        if not ret: # Nếu không còn khung hình (hết video)

            break


        frame = detector.detect_frame(frame) # Xử lý khung hình bằng detector

        # Mã hóa khung hình thành JPEG với chất lượng 80% để tăng tốc độ

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if not ret: # Nếu mã hóa thất bại

            continue # Bỏ qua khung hình này


        frame_bytes = buffer.tobytes() # Chuyển buffer thành bytes

        # Tạo response multipart cho streaming

        yield (b'--frame\r\n'

        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


    cap.release() # Giải phóng video sau khi xử lý xong

    if os.path.exists(video_path): # Kiểm tra file video còn tồn tại không

        os.remove(video_path) # Xóa file video để tiết kiệm bộ nhớ


def yolo(): # Hàm xử lý route '/'

    form = UploadForm() # Tạo instance của UploadForm

    if form.validate_on_submit(): # Nếu form được submit và hợp lệ

        video = request.files.get("file") # Lấy file video từ request

        filename = secure_filename(video.filename) # Bảo mật tên file

        video_path = os.path.join(Config.VIDEO_FOLDER, filename) # Tạo đường dẫn lưu file

        video.save(video_path) # Lưu file video vào thư mục upload

        # Trả về response streaming với các khung hình đã xử lý

        return Response(generate_frames(video_path),mimetype='multipart/x-mixed-replace; boundary=frame')

    # Nếu không submit, hiển thị trang HTML với form

    return render_template('yolo.html', form=form)
