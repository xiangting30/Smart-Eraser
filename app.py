import cv2
import numpy as np
import os
import shutil
import uuid
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/results'

# 确保必要的文件夹都存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 用于存储处理进度 (实际生产环境中建议使用Redis或数据库)
progress_dict = {}

def get_video_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        preview_path = video_path + "_preview.jpg"
        cv2.imwrite(preview_path, frame)
        cap.release()
        return preview_path
    cap.release()
    return None

def process_image(path, rois, task_id):
    progress_dict[task_id] = 10  # 图片处理较快，直接给个初始进度
    img = cv2.imread(path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # 遍历多个框选区域并绘制遮罩
    for roi in rois:
        x, y, w, h = roi
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
    progress_dict[task_id] = 50
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], "result_" + os.path.basename(path))
    cv2.imwrite(output_path, result)
    
    progress_dict[task_id] = 100
    return output_path

def process_video(path, rois, task_id):
    progress_dict[task_id] = 0
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], "result_" + os.path.basename(path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    ret, frame = cap.read()
    if not ret: return None
    
    # 初始化多个追踪器
    trackers = []
    for roi in rois:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(roi))
        trackers.append(tracker)
        
    frame_count = 0
    while ret:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # 更新每个框的追踪状态
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                ix, iy, iw, ih = [int(v) for v in bbox]
                # 边界保护
                ix, iy = max(0, ix), max(0, iy)
                cv2.rectangle(mask, (ix, iy), (ix + iw, iy + ih), 255, -1)
                
        clean_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        out.write(clean_frame)
        
        frame_count += 1
        # 计算并更新进度
        if total_frames > 0:
            progress_dict[task_id] = int((frame_count / total_frames) * 100)
            
        ret, frame = cap.read()
        
    cap.release()
    out.release()
    progress_dict[task_id] = 100
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({"error": "No file"})
    file = request.files['file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    preview_url = ""
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        preview_path = get_video_first_frame(save_path)
        if preview_path:
            static_preview = os.path.join(app.config['OUTPUT_FOLDER'], filename + ".jpg")
            shutil.move(preview_path, static_preview)
            preview_url = "/" + static_preview
    return jsonify({"path": save_path, "preview_url": preview_url})

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    # 返回当前进度百分比
    return jsonify({"progress": progress_dict.get(task_id, 0)})

@app.route('/remove', methods=['POST'])
def remove():
    data = request.json
    path = data['path']
    rois = data['rois']  # 改为接收列表
    task_id = data['task_id']
    is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))
    res_path = process_video(path, rois, task_id) if is_video else process_image(path, rois, task_id)
    return jsonify({"result_url": "/" + res_path})

# 【关键】下面这部分必须存在且在最底部
if __name__ == '__main__':
    import os
    # 这一步是让 Render 自动分配端口
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
