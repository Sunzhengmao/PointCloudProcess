'''
2020-07-09
孙正茂
'''
import base64, os

base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, "image")
name = "8-1.jpg"
image = os.path.join(image_path, name)
f=open(image,'rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
print(ls_f)