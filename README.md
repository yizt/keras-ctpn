# keras-ctpn

[TOC]

1. [说明](#说明)
2. [预测](#预测)
3. [训练](#训练)
4. [例子](#例子)
5. [toDoList](#toDoList)

## 说明

​         本工程是keras实现的[CPTN: Detecting Text in Natural Image with Connectionist Text Proposal Network](https://arxiv.org/abs/1609.03605) . 本工程实现主要参考了[keras-faster-rcnn](https://github.com/yizt/keras-faster-rcnn) ; 并在ICDAR2015和ICDAR2017数据集上训练和测试。

​         工程地址: [keras-ctpn](https://github.com/yizt/keras-ctpn)

​         cptn论文翻译:[CTPN.md](https://github.com/yizt/cv-papers/blob/master/CTPN.md)



关键点说明:

a.骨干网络使用的是resnet50

b.训练输入图像大小为608*608; 将图像的长边缩放到608,保持长宽比,短边padding

c.batch_size 为4, 每张图像训练256个anchor,正负样本比为1:1

d.分类、边框回归以及侧边改善的损失函数权重为1:1:1;原论文中是1:1:2

e.侧边改善与边框回归选择一样的正样本anchor;原文中应该是分开选择的

f. 侧边改善还是有效果的(注:网上很多人说没有啥效果)



## 预测

a. 工程下载

```bash
git clone https://github.com/yizt/keras-ctpn
```



b. 预训练模型下载

​    ICDAR2015训练集上训练好的模型下载地址：[ctpn.h5](https://pan.baidu.com/s/10LAyfh2pvE_ljXMYLe9gFw) 提取码:7vr8

​    ICDAR2017训练集上训练好的模型下载地址：[ctpn.050.h5](https://pan.baidu.com/s/1Wn0MY8DJqEwQyVcw0MH7Uw) 提取码:oy15

c.修改配置类config.py中如下属性

```python
	WEIGHT_PATH = '/tmp/ctpn.h5'
```

d. 检测文本

```shell
python predict.py --image_path image_3.jpg
```

## 评估

```shell
python evaluate.py --weight_path /tmp/ctpn.140.h5 --image_dir /opt/dataset/OCR/ICDAR_2015/test_images/ --output_dir /tmp/output_2015/
```

## 训练

a. 训练数据下载

 icdar2015下载地址(官网打开太慢): https://download.csdn.net/download/moonshapedpool/10645292

```shell
#icdar2017
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_training_images_1~8.zip
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_training_localization_transcription_gt_v2.zip
wget -c -t 0 http://datasets.cvc.uab.es/rrc/ch8_test_images.zip
```



b. resnet50与训练模型下载

```shell
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```



c. 修改配置类config.py中，如下属性

```python
	# 预训练模型
    PRE_TRAINED_WEIGHT = '/opt/pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # 数据集路径
    IMAGE_DIR = '/opt/dataset/OCR/ICDAR_2015/train_images'
    IMAGE_GT_DIR = '/opt/dataset/OCR/ICDAR_2015/train_gt'
```

d.训练

```shell
python train.py --epochs 50
```





## 例子

### ICDAR2015

#### 带侧边改善

![](image_examples/icdar2015/img_8.1.jpg)

![](image_examples/icdar2015/img_200.1.jpg)

#### 不带侧边改善
![](image_examples/icdar2015/img_8.0.jpg)

![](image_examples/icdar2015/img_200.0.jpg)


### ICDAR2017


![](image_examples/icdar2017/ts_img_01000.1.jpg)

![](image_examples/icdar2017/ts_img_01001.1.jpg)



## toDoList

1. 侧边改善(已完成)
2. ICDAR2017数据集训练
3. 检测文本行坐标映射到原图
4. 精度评估