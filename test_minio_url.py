import os
from minio import Minio
from minio.error import S3Error

# 1. 初始化客户端
client = Minio(
    "minio.di.qihoo.net:9000",
    access_key="zhangshuhao",
    secret_key="MinIO@2025.qihoo",
    secure=False
)

bucket_name = "facerun-content-detect"
file_path = "data/face/五月天阿信/陈信宏_1.png"   # 本地文件（jpg / jpeg / png）

# 2. 支持的图片类型 & Content-Type 映射
ALLOWED_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

def upload_image(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 不存在")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXT:
        raise ValueError(f"不支持的图片格式: {ext}")

    content_type = ALLOWED_EXT[ext]
    object_name = f"image/{filename}"

    # 建桶（若不存在）
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    # 上传
    client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=file_path,
        content_type=content_type
    )

    url = f"http://minio.di.qihoo.net:9000/{bucket_name}/{object_name}"
    return url


try:
    url = upload_image(file_path)
    print(f"✅ 上传成功: {url}")
except (S3Error, Exception) as e:
    print("❌ 上传失败:", e)

