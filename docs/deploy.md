# Strongeryolo-pytorch 

## Deployment on mobile devices
Pytorch is an awesome framework to validate ideas, but currently not easy to deploy on mobile devices. So I use tensorflow as a **Serialization Tool** to port a static graph model. Then deploy the model with tflite,MNN,NPU...

## Prepare the environments
1 . Install tensorflow1.13(cpu version will just do fine) alongside with pytorch.  
2 . Install [MNN-python](https://www.yuque.com/mnn/cn/dmqa3z) alongside with pytorch.
```
conda install tensorflow==1.13
pip install  MNN
```
## Usage
1 . Download the corresponding checkpoint here([Link](https://pan.baidu.com/s/17VK455rp4B_SRhEmklT_ig) password: i3pa).The pruned model is support and recommended :).   
2 . Directly run the demo.    
```
python deploy/port2mnn.py  
``` 
3 . See another [MNN-demo](https://github.com/wlguan/MNN-yolov3) for cpp and android deployment.
