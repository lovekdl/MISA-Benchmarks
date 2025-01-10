import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 基础配置信息
BASE_URL = "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en"
DOWNLOAD_DIR = "/root/workspace/cthu1g4p420c73egakcg/BCD-Benchmarks/datasets/C4/en"

VALIDATION_COUNT = 8
TRAIN_COUNT = 1024
MAX_WORKERS = 8

def download_file(file_url: str, output_dir: str):
    """
    调用 wget 进行下载
    可以在此处添加 "-c" 参数实现断点续传，例如：
    subprocess.run(["wget", "-c", "--directory-prefix", output_dir, file_url], check=True)
    """
    subprocess.run(["wget", "-c", "--directory-prefix", output_dir, file_url], check=True)

def main():
    urls = []

    for i in range(VALIDATION_COUNT):
        file_name = f"c4-validation.{i:05d}-of-00008.json.gz"
        urls.append(f"{BASE_URL}/{file_name}")

    for i in range(TRAIN_COUNT):
        file_name = f"c4-train.{i:05d}-of-01024.json.gz"
        urls.append(f"{BASE_URL}/{file_name}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {
            executor.submit(download_file, url, DOWNLOAD_DIR): url
            for url in urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
                print(f"[SUCCESS] {url}")
            except Exception as exc:
                print(f"[ERROR] Failed to download {url}. Reason: {exc}")

main()
