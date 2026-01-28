"""
测试用例 - 测试 JianceAI Audit Agent API
"""

import requests
import time
import json
from pathlib import Path


class TestAuditAgentAPI:
    """测试审核智能体 API"""
    
    BASE_URL = "http://127.0.0.1:8001"
    
    def test_health_check(self):
        """测试健康检查接口"""
        print("\n" + "="*50)
        print("测试 1: 健康检查接口")
        print("="*50)
        
        try:
            response = requests.get(f"{self.BASE_URL}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {response.json()}")
            
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
            print("✅ 健康检查测试通过")
            return True
        except Exception as e:
            print(f"❌ 健康检查测试失败: {e}")
            return False
    
    def test_analyze_with_image(self, image_path: str, enable_search: bool = False):
        """测试图片审核接口"""
        print("\n" + "="*50)
        print("测试 2: 图片审核接口")
        print("="*50)
        print(f"图片路径: {image_path}")
        print(f"启用搜索: {enable_search}")
        
        if not Path(image_path).exists():
            print(f"❌ 图片文件不存在: {image_path}")
            return False
        
        try:
            # 准备请求数据
            files = {
                'file': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
            }
            data = {
                'enable_search': str(enable_search).lower()
            }
            
            print("📤 发送请求...")
            start_time = time.time()
            
            # 发送请求并处理 SSE 流式响应
            response = requests.post(
                f"{self.BASE_URL}/analyze",
                files=files,
                data=data,
                stream=True,
                timeout=300  # 5分钟超时
            )
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ 请求耗时: {elapsed_time:.2f} 秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ 请求失败: {response.text}")
                return False
            
            # 处理 SSE 流式响应
            print("\n📥 接收流式响应:")
            print("-" * 50)
            
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f"[事件 {event_count}] {line_str}")
                    event_count += 1
            
            print("-" * 50)
            print(f"✅ 共接收到 {event_count} 个事件")
            print("✅ 图片审核测试通过")
            return True
            
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except Exception as e:
            print(f"❌ 图片审核测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 关闭文件
            if 'file' in files:
                files['file'][1].close()
    
    def test_analyze_with_video(self, video_path: str, enable_search: bool = False):
        """测试视频审核接口"""
        print("\n" + "="*50)
        print("测试 3: 视频审核接口")
        print("="*50)
        print(f"视频路径: {video_path}")
        print(f"启用搜索: {enable_search}")
        
        if not Path(video_path).exists():
            print(f"❌ 视频文件不存在: {video_path}")
            return False
        
        try:
            # 准备请求数据
            files = {
                'file': (Path(video_path).name, open(video_path, 'rb'), 'video/mp4')
            }
            data = {
                'enable_search': str(enable_search).lower()
            }
            
            print("📤 发送请求...")
            start_time = time.time()
            
            # 发送请求并处理 SSE 流式响应
            response = requests.post(
                f"{self.BASE_URL}/analyze",
                files=files,
                data=data,
                stream=True,
                timeout=600  # 10分钟超时（视频处理可能需要更长时间）
            )
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ 请求耗时: {elapsed_time:.2f} 秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ 请求失败: {response.text}")
                return False
            
            # 处理 SSE 流式响应
            print("\n📥 接收流式响应:")
            print("-" * 50)
            
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f"[事件 {event_count}] {line_str}")
                    event_count += 1
            
            print("-" * 50)
            print(f"✅ 共接收到 {event_count} 个事件")
            print("✅ 视频审核测试通过")
            return True
            
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except Exception as e:
            print(f"❌ 视频审核测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 关闭文件
            if 'file' in files:
                files['file'][1].close()
    
    def test_analyze_with_audio(self, audio_path: str, enable_search: bool = False):
        """测试音频审核接口"""
        print("\n" + "="*50)
        print("测试 4: 音频审核接口")
        print("="*50)
        print(f"音频路径: {audio_path}")
        print(f"启用搜索: {enable_search}")
        
        if not Path(audio_path).exists():
            print(f"❌ 音频文件不存在: {audio_path}")
            return False
        
        try:
            # 准备请求数据
            files = {
                'file': (Path(audio_path).name, open(audio_path), 'audio/mpeg')
            }
            data = {
                'enable_search': str(enable_search).lower()
            }
            
            print("📤 发送请求...")
            start_time = time.time()
            
            # 发送请求并处理 SSE 流式响应
            response = requests.post(
                f"{self.BASE_URL}/analyze",
                files=files,
                data=data,
                stream=True,
                timeout=300  # 5分钟超时
            )
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ 请求耗时: {elapsed_time:.2f} 秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ 请求失败: {response.text}")
                return False
            
            # 处理 SSE 流式响应
            print("\n📥 接收流式响应:")
            print("-" * 50)
            
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f"[事件 {event_count}] {line_str}")
                    event_count += 1
            
            print("-" * 50)
            print(f"✅ 共接收到 {event_count} 个事件")
            print("✅ 音频审核测试通过")
            return True
            
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except Exception as e:
            print(f"❌ 音频审核测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 关闭文件
            if 'file' in files:
                files['file'][1].close()


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("JianceAI Audit Agent API 测试套件")
    print("="*60)
    
    tester = TestAuditAgentAPI()
    
    # 测试结果统计
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0
    }
    
    # 1. 测试健康检查
    results["total"] += 1
    if tester.test_health_check():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # 2. 测试图片审核（需要提供实际的图片路径）
    # 请替换为实际的图片路径
    test_image = r"C:\Users\maxiaoguang\Pictures\2841_449752_652827.jpg"
    if Path(test_image).exists():
        results["total"] += 1
        if tester.test_analyze_with_image(test_image, enable_search=True):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # 3. 测试视频审核（需要提供实际的视频路径）
    # 请替换为实际的视频路径
    test_video = r"C:\Users\maxiaoguang\Downloads\79cf6cfdb6d7d3ab8f42e00903a09d1e.mp4"
    if Path(test_video).exists():
        results["total"] += 1
        if tester.test_analyze_with_video(test_video, enable_search=True):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # 4. 测试音频审核（需要提供实际的音频路径）
    # 请替换为实际的音频路径
    test_audio = "test_data/test_audio.mp3"
    if Path(test_audio).exists():
        results["total"] += 1
        if tester.test_analyze_with_audio(test_audio, enable_search=False):
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # 打印测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"总测试数: {results['total']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"通过率: {results['passed']/results['total']*100:.1f}%" if results['total'] > 0 else "N/A")
    print("="*60)


if __name__ == "__main__":
    # 创建测试数据目录
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("\n📝 提示:")
    print("1. 请确保服务已启动: uvicorn judge_agent.main:app --reload")
    print("2. 请将测试文件放入 test_data/ 目录:")
    print("   - test_image.jpg (测试图片)")
    print("   - test_video.mp4 (测试视频)")
    print("   - test_audio.mp3 (测试音频)")
    print("3. 或者修改代码中的文件路径")
    print("\n按 Enter 键开始测试...")
    input()
    
    # 运行测试
    main()