# keras-yolov3
源项目：https://github.com/qqwweee/keras-yolo3

本项目就是关于上面项目的个人理解。

仅想看效果的话，直接跑my_inference.py就好，

## 预训练模型下载
在 链接:https://pan.baidu.com/s/1ZHQRr57ksEYWsnav50pwkA  密码:tg07
下载所需要的模型（.weights）,并将模型放入 ./model_data/ 文件夹中。

执行下列语句进行模型转换
```
python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
```



## 推断

* 1）可以使用源项目的推断，比如：
```
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```
具体见 yolo_video.py

* 2）也可使用我的推断，具体见my_inference.py.


我写的做了简单的重构，可以传入图片，然后得到展示结果。

这样无论是对单张图片还是视频都能比较好的测试。源项目的推断代码需要将图片放到某个文件夹下，比较麻烦。

## 训练

暂时没看
## 测试

暂时没看