## 环境准备

1. python(>=3.10)
2. pip
3. [ffmpeg](https://www.youtube.com/watch?v=OlNWCpFdVMA)
4. [visual studio 2022 runtimes (windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   
## 模型下载
5. [GFPGANv1.4](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
6. [inswapper_128.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx)或者 _[另一个版本](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx)_

## 下载项目
```
git clone https://github.com/mx-pai/face_swap_live.git
```

## 安装依赖
   
   ```
   pip install -r requirements.txt
   ```
<details>
   
### CUDA用户使用
- 下载 [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- 更新依赖:
```
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu==1.16.3
```
- 运行项目
```
python run.py --execution-provider cuda
```
<details/>
   
---

引用自[开源项目](https://github.com/hacksider/Deep-Live-Cam.git)
