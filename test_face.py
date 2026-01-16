import requests
import json

url = "http://hpcinf01.aitc.bjwdt.qihoo.net:6980/api/v1/image/sync"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

data = {
    "ability": ["face"],
    "tasks": [
        #{"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/曾志伟_ref.png"}
        #{"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/周永康.jpg"}
        #{"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/郭伯雄.jpg"}
        #{"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/00202401054text00c5khinfhcvpvio40046b4c.jpg"}
        #{"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/zhaowei.jpeg"}
        {"dataId": "asdf1", "url": "http://minio.di.qihoo.net:9000/facerun-content-detect/image/bifujian.jpg"}
    ],
    "rule": []
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.json())

