# AreWeMeetBefore 🤝

**Video Face Recognition and Temporal Analysis Based on Insightface and Faiss**

![GitHub](https://img.shields.io/github/license/BaiHLiu/AreWeMeetBefore)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)


> 在这个忙碌的世界里，你是否曾经好奇，那个在咖啡厅、地铁或会议上匆匆一瞥的陌生人，也许其实是你曾经相遇过的人？
> 
> In this busy world, have you ever wondered if the stranger you glimpsed at a cafe, subway, or meeting might actually be someone you've encountered before?

基于`InsightFace`和`Faiss`向量检索技术，帮助你探索视频中的人脸时空关系，发现那些与你多次相遇却未曾相识的人。

Based on `InsightFace` and `Faiss` vector retrieval technology, this project helps you explore spatiotemporal relationships of faces in videos, discovering those


## 🚀 功能特点

1. **视频分割与帧提取**：从视频中提取时间戳帧，为人脸分析提供基础
2. **高效人脸识别**：使用InsightFace进行人脸检测和特征提取
3. **向量相似性搜索**：基于Faiss的高性能向量检索，快速匹配相似人脸
4. **时间序列分析**：分析人脸出现的时间模式，发现长时间跨度的相遇
5. **人脸数据库管理**：自动维护本地人脸库，支持增量更新
6. **可视化分析**：提供Jupyter Notebook示例，辅助探索分析结果

## 📋 安装与准备

### 前置条件

- Python 3.8+
- FFmpeg（需安装并添加到环境变量中）
- NVIDIA GPU（推荐，但不是必需）

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/yourusername/AreWeMeetBefore.git
   cd AreWeMeetBefore
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 准备视频文件
   将需要分析的视频文件放置在 video 目录下

## 🎬 使用方法

### 1. 视频分割（提取帧）

```bash
python video_spliter.py
```

默认会处理 video 目录下的所有MP4文件，并将提取的帧保存到 splitted 目录。

参数说明:
- 可在代码中修改 `interval_seconds` 变量来调整抽帧间隔（默认为1秒）

### 2. 人脸检测与识别

```bash
python batch_face_processor.py
```

默认会处理 splitted 目录下的所有图片，进行人脸检测和识别，结果保存到 db 目录。

高级选项:
```bash
python batch_face_processor.py --input ./my_images --output ./my_faces --db ./my_db --threshold 0.65
```

参数说明：
- `--input`: 输入图片目录 (默认: ./video/splitted)
- `--output`: 输出人脸图片目录 (默认: ./faces)
- `--db`: 数据库文件目录 (默认: ./db)
- `--threshold`: 人脸匹配相似度阈值 (默认: 0.60)

### 3. 时间序列分析

使用提供的Jupyter Notebook进行分析:

```bash
jupyter notebook analyse.ipynb
```

分析思路:
- 核心算法旨在寻找与自己多次相遇且时间跨度最大的人
- 根据出现频次和时间跨度计算相遇显著性
- 可以自由探索修改算法以满足不同的分析需求

## 📁 目录结构

```
./
├── video_spliter.py         # 视频分割工具
├── batch_face_processor.py  # 人脸处理主程序
├── analyse.ipynb            # 时间序列分析示例
├── video/                   # 视频相关目录
│   └── splitted/            # 拆分后的图片（输入目录）
├── faces/                   # 保存的人脸图片（输出目录）
└── db/                      # 数据库文件目录
    ├── face_index.bin       # Faiss索引文件
    ├── face_metadata.jsonl  # 人脸元数据JSONL文件
    └── failed_images.txt    # 处理失败的图片记录
```

## 📊 数据格式

face_metadata.jsonl中每行是一个JSON对象，表示一个人脸记录，格式如下：

```json
{
  "face_id": 0,
  "first_seen": "20250307122924.jpg",
  "last_seen": "20250307123008.jpg",
  "appearances": ["20250307122924.jpg", "20250307123005.jpg", "20250307123008.jpg"],
  "count": 3,
  "face_image": "face_0.jpg"
}
```

字段说明：
- `face_id`: 人脸唯一标识符，与Faiss索引对应
- `first_seen`: 首次出现的图片文件名（时间戳）
- `last_seen`: 最近一次出现的图片文件名（时间戳）
- `appearances`: 出现过的所有图片文件名列表
- `count`: 出现次数
- `face_image`: 对应的保存的人脸图片文件名

## 📝 注意事项

1. 视频文件名格式默认为DJI命名规则（如：DJI_20250307122924_0036_D.MP4），其中包含时间戳信息，若需修改正则表达式，请前往`video_spliter.py`修改`VIDEO_FILENAME_REGEX`。
2. 相似度阈值越高，区分不同人脸的能力越强，但可能导致同一个人被识别为多个不同的ID。
3. 处理失败的图片会被记录到`failed_images.txt`文件中。
4. 硬件配置会影响处理速度，推荐使用GPU以加速处理。
5. 使用本工具时，请尊重他人隐私。
6. 本工具包含*AIGC*提供的内容，请注意甄别。

## 🔄 工作流程

1. 视频分割：提取视频帧并保存为时间戳命名的图片。
2. 人脸处理：检测图片中的人脸，提取特征向量并匹配已有人脸。
3. 时间分析：基于人脸出现的时间戳进行时序分析，发现相遇模式。
4. 结果探索：通过Jupyter Notebook交互式探索分析结果。

## 📄 许可证

MIT License

## 🙏 鸣谢

- [InsightFace](https://github.com/deepinsight/insightface) - 提供高精度人脸识别算法
- [Faiss](https://github.com/facebookresearch/faiss) - 高性能向量相似性搜索
- [FFmpeg](https://ffmpeg.org/) - 视频处理支持

---

💡 **探索你的时空相遇，发现被忽略的缘分。**
