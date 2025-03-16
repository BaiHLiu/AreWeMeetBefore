import cv2
import numpy as np
import insightface
import faiss
import os
import json
import argparse
from tqdm import tqdm
import time
from datetime import datetime
from insightface.app import FaceAnalysis

class BatchFaceProcessor:
    def __init__(self, 
                 input_dir="./video/splitted",
                 output_dir="./faces",
                 db_dir="./db",
                 similarity_threshold=0.60,
                 model_path='/Users/catop/.insightface/models/buffalo_l/w600k_r50.onnx'):

        # 创建必要的目录
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.db_dir = db_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # 设置相似度阈值
        self.similarity_threshold = similarity_threshold
        
        # 初始化人脸识别模型
        self.setup_face_model()
        
        # 初始化Faiss索引
        self.vector_dim = 512  # InsightFace生成的特征向量维度
        self.index_file = os.path.join(self.db_dir, "face_index.bin")
        self.metadata_file = os.path.join(self.db_dir, "face_metadata.jsonl")
        
        # 加载或创建Faiss索引和metadata
        self.load_or_create_database()
        
        # 记录处理失败的图片
        self.failed_images_file = os.path.join(self.db_dir, "failed_images.txt")
        
    def setup_face_model(self):
        """设置人脸识别模型"""
        self.handler = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.handler.prepare(ctx_id=0, det_size=(640, 640))
    
    def load_or_create_database(self):
        """加载现有数据库或创建新的数据库"""
        # 尝试加载Faiss索引
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"已加载Faiss索引，包含 {self.index.ntotal} 条记录")
        else:
            self.index = faiss.IndexFlatL2(self.vector_dim)
            print("已创建新的Faiss索引")
        
        # 尝试加载metadata
        self.face_metadata = []
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        self.face_metadata.append(json.loads(line.strip()))
            print(f"已加载元数据，包含 {len(self.face_metadata)} 条记录")
            
            # 验证Faiss索引和metadata是否一致
            if self.index.ntotal != len(self.face_metadata):
                print(f"警告: Faiss索引数量 ({self.index.ntotal}) 与元数据数量 ({len(self.face_metadata)}) 不一致！")
        else:
            print("已创建新的元数据文件")
    
    def save_database(self):
        """保存Faiss索引和元数据"""
        # 保存Faiss索引
        faiss.write_index(self.index, self.index_file)
        print(f"Faiss索引已保存，包含 {self.index.ntotal} 条记录")
    
    def append_metadata(self, metadata_entry):
        """追加单条元数据到jsonl文件"""
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata_entry, ensure_ascii=False) + '\n')
    
    def update_metadata_file(self, idx):
        """更新jsonl文件中的特定记录"""
        if idx >= len(self.face_metadata):
            print(f"错误: 索引 {idx} 超出元数据范围")
            return
        
        # 使用临时文件来更新特定行
        temp_file = self.metadata_file + '.temp'
        with open(self.metadata_file, 'r', encoding='utf-8') as f_in, \
             open(temp_file, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in):
                if i == idx:
                    # 替换这一行
                    f_out.write(json.dumps(self.face_metadata[idx], ensure_ascii=False) + '\n')
                else:
                    f_out.write(line)
        
        # 用临时文件替换原文件
        os.replace(temp_file, self.metadata_file)
    
    def record_failed_image(self, image_path, error_msg):
        """记录处理失败的图片"""
        with open(self.failed_images_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {image_path} - {error_msg}\n")
    
    def extract_face_features(self, image_path):
        """从图像中提取人脸特征"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.record_failed_image(image_path, "无法读取图像")
                return None
            
            faces = self.handler.get(img)
            if len(faces) == 0:
                self.record_failed_image(image_path, "未检测到人脸")
                return None
                
            return img, faces
        except Exception as e:
            self.record_failed_image(image_path, f"处理异常: {str(e)}")
            return None
    
    def search_similar_face(self, face_embedding):
        """在Faiss索引中搜索相似人脸"""
        if self.index.ntotal == 0:
            return None, -1  # 数据库为空
        
        # 执行搜索
        distances, indices = self.index.search(face_embedding.reshape(1, -1).astype('float32'), 1)
        distance = distances[0][0]
        idx = indices[0][0]
        
        # 计算相似度分数 (1.0 / (1.0 + distance))
        similarity = 1.0 / (1.0 + distance)
        
        if similarity >= self.similarity_threshold:
            return similarity, idx
        else:
            return similarity, -1  # 相似度不够
    
    def crop_and_save_face(self, img, face, face_id):
        """裁剪并保存人脸图像"""
        bbox = face.bbox.astype(int)
        # 增加一些边距
        margin = 30
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(img.shape[1], bbox[2] + margin)
        y2 = min(img.shape[0], bbox[3] + margin)
        
        face_img = img[y1:y2, x1:x2]
        face_filename = os.path.join(self.output_dir, f"face_{face_id}.jpg")
        cv2.imwrite(face_filename, face_img)
        return face_filename
    
    def process_image(self, image_path):
        """处理单张图像"""
        result = self.extract_face_features(image_path)
        if result is None:
            return
        
        img, faces = result
        image_filename = os.path.basename(image_path)  # 使用文件名作为时间标识
        
        for face_idx, face in enumerate(faces):
            embedding = face.normed_embedding
            similarity, match_idx = self.search_similar_face(embedding)
            
            if match_idx >= 0:
                # 人脸已存在，更新记录
                self.face_metadata[match_idx]["appearances"].append(image_filename)
                self.face_metadata[match_idx]["last_seen"] = image_filename
                self.face_metadata[match_idx]["count"] += 1
                
                # 更新jsonl文件中的记录
                self.update_metadata_file(match_idx)
                
                # 确保有对应的面部图像文件
                if not os.path.exists(os.path.join(self.output_dir, f"face_{match_idx}.jpg")):
                    self.crop_and_save_face(img, face, match_idx)
            else:
                # 添加新人脸
                face_id = self.index.ntotal  # 使用当前索引总数作为新的人脸ID
                
                # 保存人脸图像
                face_image_path = self.crop_and_save_face(img, face, face_id)
                
                # 添加到Faiss索引
                self.index.add(embedding.reshape(1, -1).astype('float32'))
                
                # 创建并保存元数据
                metadata_entry = {
                    "face_id": face_id,
                    "first_seen": image_filename,
                    "last_seen": image_filename,
                    "appearances": [image_filename],
                    "count": 1,
                    "face_image": os.path.basename(face_image_path)
                }
                
                self.face_metadata.append(metadata_entry)
                self.append_metadata(metadata_entry)
    
    def process_directory(self):
        """批量处理目录中的所有图像"""
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        # 获取所有图像文件
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"在目录 {self.input_dir} 中未找到图像文件")
            return
        
        # 排序文件名，假设文件名是按时间顺序命名的
        image_files.sort()
        
        # 使用tqdm显示进度
        for image_path in tqdm(image_files, desc="处理图像"):
            self.process_image(image_path)
        
        # 完成后保存Faiss索引
        self.save_database()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量图像人脸识别与Faiss数据库处理')
    parser.add_argument('--input', type=str, default='./video/splitted',
                       help='输入图像目录路径')
    parser.add_argument('--output', type=str, default='./faces',
                       help='输出人脸图像目录')
    parser.add_argument('--db', type=str, default='./db',
                       help='数据库目录')
    parser.add_argument('--threshold', type=float, default=0.60,
                       help='人脸匹配相似度阈值 (0-1)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    print(f"初始化批量人脸处理系统...")
    processor = BatchFaceProcessor(
        input_dir=args.input,
        output_dir=args.output,
        db_dir=args.db,
        similarity_threshold=args.threshold
    )
    
    print(f"开始批量处理图像，输入目录: {args.input}")
    print(f"人脸匹配相似度阈值: {args.threshold}")
    processor.process_directory()
    
    print("处理完成！")
    print(f"已处理的人脸总数: {processor.index.ntotal}")
    print(f"人脸图像已保存到: {args.output}")
    print(f"数据库文件已保存到: {args.db}")
    
    if os.path.exists(processor.failed_images_file):
        with open(processor.failed_images_file, 'r') as f:
            failed_count = len(f.readlines())
        print(f"处理失败的图像数: {failed_count} (详情请查看 {processor.failed_images_file})")

if __name__ == "__main__":
    main()
