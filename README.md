# PaddleDetection知识蒸馏

知识蒸馏主要是让让新模型（通常是一个参数量更少的模型）近似原模型(模型即函数)。注意到，在机器学习中，我们常常假定输入到输出有一个潜在的函数关系，这个函数是未知的：从头学习一个新模型就是从有限的数据中近似一个未知的函数。如果让新模型近似原模型，因为原模型的函数是已知的，我们可以使用很多非训练集内的伪数据来训练新模型。


# 分类模型蒸馏：
原来我们需要让新模型的softmax分布与真实标签匹配，现在只需要让新模型与原模型在给定输入下的softmax分布匹配了。但是由于softmax函数是一个约等于arg max的近似，它所能描述的知识（对输出的概率描述）非常有限，一种常用的解决方法是直接让新旧模型匹配logits输出，即使用teacher model的logits输出作为student model的回归目标，并使用L2损失作为loss。

## one-stage检测模型蒸馏：

**基本思路**

One-stage目标检测任务的训练目标难度更大，因为teacher网络会预测出更多的背景bbox，如果直接用teacher的预测输出作为student学习的soft label会有严重的类别不均衡问题。解决这个问题需要引入新的方法。

主要是《Object detection at 200 Frames Per Second》这篇文章中提出了针对该问题的解决方案，即针对YOLOv3中分类、回归、objectness三个不同的head适配不同的蒸馏损失函数，并对分类和回归的损失函数用objectness分值进行抑制，以解决前景背景类别不均衡问题。

并且该文章使用未标注数据作为蒸馏损失，跟检测损失（有标注数据）加权求和作为最终的损失函数。

![](https://ai-studio-static-online.cdn.bcebos.com/b66b22ea901f4d398e330465c7e8c0a041629aa8e82a43af8e36d757480d47e5)


# 关于作者
> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)

# 实验环境：

数据集：VOC2012

GPU：V100*1

Batch Size：8

Epoches：70000


```python
!rm -rf PaddleDetection/
```


```python
!unzip -oq data/data85000/PaddleDetection-master.zip -d ./
```


```python
%cd PaddleDetection-master/
```

    /home/aistudio/PaddleDetection-master



```python
!pip install -r requirements.txt
```


```python
!pip install paddleslim
```


```python
!mkdir /home/aistudio/PaddleDetection-master/dataset/voc/VOCdevkit
```


```python
!unzip -oq ../data/data39480/VOC2012.zip -d dataset/voc/VOCdevkit/
```

# 划分数据集


```python
import os
from tqdm import tqdm
from random import shuffle

base = 'dataset/voc/'
img_base = 'VOCdevkit/VOC2012/JPEGImages/'
xml_base = 'VOCdevkit/VOC2012/Annotations/'

images_list = os.listdir(os.path.join(base, img_base))
shuffle(images_list)

split_num = int(0.9 * len(images_list))

with open(os.path.join(base, 'trainval.txt'), 'w') as f:
    for im in tqdm(images_list[:split_num]):
        img_id = im[:-4]
        line = '{}{}.jpg {}{}.xml\n'.format(img_base, img_id, xml_base, img_id)
        f.write(line)

with open(os.path.join(base, 'test.txt'), 'w') as f:
    for im in tqdm(images_list[split_num:]):
        img_id = im[:-4]
        line = '{}{}.jpg {}{}.xml\n'.format(img_base, img_id, xml_base, img_id)
        f.write(line)
```

    100%|██████████| 15412/15412 [00:00<00:00, 915578.85it/s]
    100%|██████████| 1713/1713 [00:00<00:00, 792766.50it/s]


## 正常训练：
正常训练反而是收敛最快的。


```python
!python tools/train.py \
    -c configs/yolov3_mobilenet_v1_voc.yml --eval
```

    2021-04-30 13:02:48,341 - INFO - iter: 29200, lr: 0.001000, 'loss': '19.766043', eta: 2:01:38, batch_cost: 0.17888 sec, ips: 44.72353 images/sec


## L2蒸馏损失训练
实验结果：
Best test box ap: 55.303840240819966, in step: 69999


```python
!python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1_voc.yml \
    -t configs/yolov3_r34_voc.yml \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
```

    2021-04-29 15:50:48,397 - INFO - step 68500 lr 0.000010, loss 138.481812, distill_loss 120.378197, teacher_loss 15.744522
    2021-04-29 15:51:09,905 - INFO - step 68600 lr 0.000010, loss 160.502563, distill_loss 146.543015, teacher_loss 10.307454
    2021-04-29 15:51:29,715 - INFO - step 68700 lr 0.000010, loss 107.800804, distill_loss 96.591446, teacher_loss 11.338940
    2021-04-29 15:51:50,172 - INFO - step 68800 lr 0.000010, loss 119.590355, distill_loss 106.403137, teacher_loss 14.288318
    2021-04-29 15:52:10,472 - INFO - step 68900 lr 0.000010, loss 122.257706, distill_loss 97.294144, teacher_loss 19.427813
    2021-04-29 15:52:50,220 - INFO - step 69100 lr 0.000010, loss 144.336548, distill_loss 129.560059, teacher_loss 14.828430
    2021-04-29 15:53:09,471 - INFO - step 69200 lr 0.000010, loss 124.607025, distill_loss 112.543617, teacher_loss 12.816269
    2021-04-29 15:53:29,804 - INFO - step 69300 lr 0.000010, loss 123.180389, distill_loss 106.882095, teacher_loss 13.914233
    2021-04-29 15:53:52,323 - INFO - step 69400 lr 0.000010, loss 129.621185, distill_loss 106.162689, teacher_loss 19.629433
    2021-04-29 15:54:12,425 - INFO - step 69500 lr 0.000010, loss 127.870285, distill_loss 112.545990, teacher_loss 12.473295
    2021-04-29 15:54:32,816 - INFO - step 69600 lr 0.000010, loss 194.085114, distill_loss 181.748535, teacher_loss 12.321056
    2021-04-29 15:54:53,872 - INFO - step 69700 lr 0.000010, loss 214.038483, distill_loss 186.034805, teacher_loss 23.179140
    2021-04-29 15:55:13,472 - INFO - step 69800 lr 0.000010, loss 123.184227, distill_loss 106.268097, teacher_loss 16.398609
    2021-04-29 15:55:33,172 - INFO - step 69900 lr 0.000010, loss 91.018211, distill_loss 76.597687, teacher_loss 10.691896
    2021-04-29 15:55:52,595 - INFO - Save model to output/yolov3_mobilenet_v1_voc/model_final.
    2021-04-29 15:55:56,586 - INFO - Test iter 0
    2021-04-29 15:56:05,482 - INFO - Test iter 100
    2021-04-29 15:56:12,581 - INFO - Test iter 200
    2021-04-29 15:56:13,775 - INFO - Test finish iter 215
    2021-04-29 15:56:13,775 - INFO - Total number of images: 1713, inference time: 99.09352407371594 fps.
    2021-04-29 15:56:13,776 - INFO - Start evaluate...
    2021-04-29 15:56:14,063 - INFO - Accumulating evaluatation results...
    2021-04-29 15:56:14,077 - INFO - mAP(0.50, 11point) = 55.30%
    2021-04-29 15:56:14,077 - INFO - Save model to output/yolov3_mobilenet_v1_voc/best_model.
    2021-04-29 15:56:18,180 - INFO - Best test box ap: 55.303840240819966, in step: 69999



```python
!mkdir L2distill
!mv output/yolov3_mobilenet_v1_voc/model_final.* L2distill
```

## Fine-gained蒸馏损失训练

《Object detection at 200 Frames Per Second》
好家伙，这么复杂结果还不如上一个L2损失。

实验结果：
Best test box ap: 44.40395830547559, in step: 66000

## 注：
该方法需要两个模型的回归目标、分类目标一一对应，因此需要观察模型输出并手动给出对应pair，因此目前只支持yolo。


```python
!python slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1_voc.yml -o use_fine_grained_loss=true\
    -t configs/yolov3_r34_voc.yml \
    --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar \
    -r output/yolov3_mobilenet_v1_voc/38000
```

    2021-04-30 11:14:19,128 - INFO - step 67200 lr 0.000010, loss 42.511688, distill_loss 12.518637, teacher_loss 22.399191
    2021-04-30 11:14:40,227 - INFO - step 67300 lr 0.000010, loss 24.595768, distill_loss 17.012951, teacher_loss 7.210430
    2021-04-30 11:15:00,549 - INFO - step 67400 lr 0.000010, loss 33.262058, distill_loss 11.598403, teacher_loss 18.617779
    2021-04-30 11:15:20,747 - INFO - step 67500 lr 0.000010, loss 33.465805, distill_loss 16.841127, teacher_loss 11.957682
    2021-04-30 11:15:43,162 - INFO - step 67600 lr 0.000010, loss 34.732750, distill_loss 22.962461, teacher_loss 11.200234
    2021-04-30 11:16:04,027 - INFO - step 67700 lr 0.000010, loss 48.137634, distill_loss 21.632807, teacher_loss 25.653572
    2021-04-30 11:16:24,828 - INFO - step 67800 lr 0.000010, loss 28.196875, distill_loss 16.726215, teacher_loss 7.880371
    2021-04-30 11:16:45,435 - INFO - step 67900 lr 0.000010, loss 33.850609, distill_loss 17.687098, teacher_loss 14.932596
    2021-04-30 11:17:07,266 - INFO - step 68000 lr 0.000010, loss 53.355064, distill_loss 31.784113, teacher_loss 15.080336
    2021-04-30 11:17:07,267 - INFO - Save model to output/yolov3_mobilenet_v1_voc/68000.
    2021-04-30 11:17:11,296 - INFO - Test iter 0
    2021-04-30 11:17:19,536 - INFO - Test iter 100
    2021-04-30 11:17:27,316 - INFO - Test iter 200
    2021-04-30 11:17:28,139 - INFO - Test finish iter 215
    2021-04-30 11:17:28,139 - INFO - Total number of images: 1713, inference time: 100.96811715424725 fps.
    2021-04-30 11:17:28,140 - INFO - Start evaluate...
    2021-04-30 11:17:28,900 - INFO - Accumulating evaluatation results...
    2021-04-30 11:17:28,942 - INFO - mAP(0.50, 11point) = 43.77%
    2021-04-30 11:17:28,945 - INFO - Best test box ap: 44.40395830547559, in step: 66000
    2021-04-30 11:17:49,528 - INFO - step 68100 lr 0.000010, loss 27.369240, distill_loss 16.653849, teacher_loss 9.920864
    2021-04-30 11:18:11,643 - INFO - step 68200 lr 0.000010, loss 33.032845, distill_loss 9.333175, teacher_loss 23.038385
    2021-04-30 11:18:32,545 - INFO - step 68300 lr 0.000010, loss 27.413500, distill_loss 15.899431, teacher_loss 8.539825
    2021-04-30 11:18:53,827 - INFO - step 68400 lr 0.000010, loss 21.284683, distill_loss 11.928459, teacher_loss 9.073975
    2021-04-30 11:19:14,828 - INFO - step 68500 lr 0.000010, loss 27.378477, distill_loss 11.061569, teacher_loss 15.730307
    2021-04-30 11:19:36,127 - INFO - step 68600 lr 0.000010, loss 19.714718, distill_loss 11.936641, teacher_loss 7.292910
    2021-04-30 11:19:57,737 - INFO - step 68700 lr 0.000010, loss 48.085701, distill_loss 31.413290, teacher_loss 13.044544
    2021-04-30 11:20:16,128 - INFO - step 68800 lr 0.000010, loss 51.778694, distill_loss 17.455339, teacher_loss 24.325071
    2021-04-30 11:20:41,966 - INFO - step 68900 lr 0.000010, loss 42.863945, distill_loss 30.657480, teacher_loss 15.654951
    2021-04-30 11:21:03,053 - INFO - step 69000 lr 0.000010, loss 27.961176, distill_loss 12.550332, teacher_loss 14.125258
    2021-04-30 11:21:22,872 - INFO - step 69100 lr 0.000010, loss 43.735428, distill_loss 19.787128, teacher_loss 16.448784
    2021-04-30 11:21:43,249 - INFO - step 69200 lr 0.000010, loss 28.012207, distill_loss 14.070652, teacher_loss 10.715887
    2021-04-30 11:22:05,127 - INFO - step 69300 lr 0.000010, loss 38.618095, distill_loss 18.164053, teacher_loss 17.802208
    2021-04-30 11:22:25,139 - INFO - step 69400 lr 0.000010, loss 48.910019, distill_loss 23.584068, teacher_loss 18.871468
    2021-04-30 11:22:46,030 - INFO - step 69500 lr 0.000010, loss 29.586151, distill_loss 12.643631, teacher_loss 15.415883
    2021-04-30 11:23:07,627 - INFO - step 69600 lr 0.000010, loss 33.945679, distill_loss 17.252405, teacher_loss 14.374951
    2021-04-30 11:23:28,778 - INFO - step 69700 lr 0.000010, loss 33.225529, distill_loss 12.227975, teacher_loss 16.934584
    2021-04-30 11:23:49,270 - INFO - step 69800 lr 0.000010, loss 30.932293, distill_loss 16.380510, teacher_loss 13.527035
    2021-04-30 11:24:10,253 - INFO - step 69900 lr 0.000010, loss 25.630695, distill_loss 13.440517, teacher_loss 9.713081
    2021-04-30 11:24:32,141 - INFO - Save model to output/yolov3_mobilenet_v1_voc/model_final.
    2021-04-30 11:24:35,957 - INFO - Test iter 0
    2021-04-30 11:24:44,138 - INFO - Test iter 100
    2021-04-30 11:24:52,015 - INFO - Test iter 200
    2021-04-30 11:24:52,835 - INFO - Test finish iter 215
    2021-04-30 11:24:52,836 - INFO - Total number of images: 1713, inference time: 100.76612474202045 fps.
    2021-04-30 11:24:52,837 - INFO - Start evaluate...
    2021-04-30 11:24:53,403 - INFO - Accumulating evaluatation results...
    2021-04-30 11:24:53,466 - INFO - mAP(0.50, 11point) = 42.63%
    2021-04-30 11:24:53,471 - INFO - Best test box ap: 44.40395830547559, in step: 66000



```python
!mkdir FineGaineddistill
!mv output/yolov3_mobilenet_v1_voc/model_final.* FineGaineddistill
```


```python
!rm -rf output/
```

# 我的公众号：

![](https://ai-studio-static-online.cdn.bcebos.com/a8e6df2fbf2646bda546f96080faa3e98fe7c58f9d194cfea440764d41bd86e5)



```python

```
