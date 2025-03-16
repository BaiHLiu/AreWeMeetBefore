import os
import re
import subprocess
from datetime import datetime, timedelta

# 定义视频文件提取的正则表达式
VIDEO_FILENAME_REGEX = r'DJI_(\d{14})_'

def extract_timestamp_from_filename(filename):
    """从DJI视频文件名中提取时间戳"""
    match = re.search(VIDEO_FILENAME_REGEX, filename)
    if match:
        timestamp_str = match.group(1)
        try:
            # 将时间戳字符串解析为datetime对象
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            return timestamp
        except ValueError:
            print(f"无法解析时间戳: {timestamp_str}")
            return None
    else:
        print(f"文件名中未找到时间戳: {filename}")
        return None

def split_video_ffmpeg(video_path, output_dir, interval_seconds=1):
    """
    使用ffmpeg按指定时间间隔分割视频并保存为jpg图像
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        interval_seconds: 分割间隔秒数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件名并提取时间戳
    video_filename = os.path.basename(video_path)
    start_timestamp = extract_timestamp_from_filename(video_filename)
    
    if not start_timestamp:
        print(f"无法从{video_filename}提取时间戳，跳过处理")
        return
    
    # 构建ffmpeg命令
    # 使用-vf fps=1/N参数来每N秒提取一帧
    cmd = [
        'ffmpeg',
        '-hwaccel', 'videotoolbox',  # 使用硬件加速
        '-i', video_path,
        '-vf', f'fps=1/{interval_seconds}',
        '-q:v', '1',  # 高质量
        '-f', 'image2',
        '-y'  # 覆盖已有文件
    ]
    
    # 获取视频持续时间（以秒为单位）
    duration_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    
    try:
        duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
        print(f"视频时长: {duration:.2f}秒")
        
        # 计算总帧数
        total_frames = int(duration / interval_seconds) + 1
        
        # 创建一个临时目录用于存储按ffmpeg默认命名的文件
        temp_dir = os.path.join(output_dir, "temp_" + os.path.splitext(video_filename)[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        # 将输出指向临时目录
        output_pattern = os.path.join(temp_dir, '%04d.jpg')
        cmd.append(output_pattern)
        
        # 执行ffmpeg命令
        print(f"正在执行: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # 重命名文件为正确的时间戳格式
        generated_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
        
        for i, file in enumerate(generated_files):
            # 计算当前帧对应的时间戳
            frame_timestamp = start_timestamp + timedelta(seconds=i * interval_seconds)
            new_filename = frame_timestamp.strftime('%Y%m%d%H%M%S') + '.jpg'
            
            # 移动并重命名文件
            old_path = os.path.join(temp_dir, file)
            new_path = os.path.join(output_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"已保存: {new_filename}")
        
        # 删除临时目录
        os.rmdir(temp_dir)
        print(f"视频 {video_filename} 处理完成，共保存 {len(generated_files)} 帧")
        
    except subprocess.CalledProcessError as e:
        print(f"处理视频时出错: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

def process_all_videos(input_dir='./video', output_dir='./video/splitted', interval_seconds=1):
    """处理指定目录下的所有MP4视频文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有MP4文件
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.MP4') or f.endswith('.mp4')]
    
    if not video_files:
        print(f"在 {input_dir} 中未找到MP4文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"正在处理: {video_file}")
        split_video_ffmpeg(video_path, output_dir, interval_seconds)

if __name__ == "__main__":
    # 可以在这里调整间隔秒数
    interval_seconds = 1
    print(f"开始处理视频文件，每 {interval_seconds} 秒提取一帧...")
    process_all_videos(interval_seconds=interval_seconds)
    print("所有视频处理完成")