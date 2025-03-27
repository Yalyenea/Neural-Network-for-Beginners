import sys, os
sys.path.append(os.pardir)  # 添加父目录到搜索路径
sys.path.append(os.path.join(os.pardir, 'dataset'))  # 明确添加dataset目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录

import numpy as np
try:
    from dataset.mnist import load_mnist
except ImportError:
    # 如果按照相对路径导入失败，尝试使用绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    try:
        from dataset.mnist import load_mnist
    except ImportError:
        print("无法导入MNIST数据集，请确保dataset目录在正确的位置")
        print("当前搜索路径:", sys.path)
        sys.exit(1)
        
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 尝试加载MNIST数据集
try:
    (a_train, b_train), (a_test, b_test) = load_mnist(flatten=True, normalize=False)
    print("MNIST数据集加载成功!")
except Exception as e:
    print(f"加载MNIST数据集时出错: {e}")
    sys.exit(1)

img = a_train[0]
label = b_train[0]
print(f"标签: {label}")  # 5

print(f"图像形状: {img.shape}")  # (784,)
img = img.reshape(28, 28)
print(f"重塑后形状: {img.shape}")  # (28, 28)

img_show(img)

print("程序执行完毕")
