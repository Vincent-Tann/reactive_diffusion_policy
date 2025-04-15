import zipfile
import os

def unzip_all(zip_dir):
    # 遍历目标目录下所有文件
    for filename in os.listdir(zip_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(zip_dir, filename)
            folder_name = filename[:-4]  # 去掉 .zip 后缀
            # extract_path = os.path.join(zip_dir, folder_name)
            extract_path = zip_dir

            # 创建解压目录（如果不存在）
            os.makedirs(extract_path, exist_ok=True)

            print(f"Unzipping {filename} -> {folder_name}/")

            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    print("✅ All zip files extracted.")

# 使用示例
if __name__ == "__main__":
    zip_directory = "/home/txs/Code/tactile/reactive_diffusion_policy/reactive_diffusion_policy_dataset/dataset_mini"  # 替换为你的路径
    unzip_all(zip_directory)
