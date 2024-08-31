# 实验记录
这是新的Fastinst实验记录，基于“统计实时样本频率重加权”的单阶段长尾实例分割实验的专用实验记录。

## 文件映射路径
>本地文件路径：D:\DeepL\detectron2-main-2024831\detectron2-main-20240831

> 远程文件路径：/ai/zhanghe/new-detectron2-for-fastinst


## 数据集
> 创建软连接，方便后续实验:
```bash 
cd datasets
ln -s /ai/zhanghe/mmdetection-3.3/data/coco .
```
annotations -> /ai/zhanghe/mmdetection-3.3/data/coco/annotations

coco -> /ai/zhanghe/mmdetection-3.3/data/coco

test2017 -> /ai/zhanghe/mmdetection-3.3/data/coco/test2017

train2017 -> /ai/zhanghe/mmdetection-3.3/data/coco/train2017

unlabeled2017 -> /ai/zhanghe/mmdetection-3.3/data/coco/unlabeled2017

val2017 -> /ai/zhanghe/mmdetection-3.3/data/coco/val2017

# 环境编译
```bash
git clone https://github.com/facebookresearch/detectron2.git
```
然后把fastinst,configs等目录拷贝到detectron2/下,执行：
```bash
python -m pip install -e detectron2
```

## 训练命令
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 --config-file configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x1_576.yaml

python train_net.py --num-gpus 6 --config-file config_path MODEL.WEIGHTS /path/to/resnet50d_ra2-464e36ba.pkl
# 实际测试：
python train_net.py --num-gpus 6 --config-file configs/COCO-instance-segmentation-for6gpu-forme/fastinst_R50-vd-dcn_ppm-fpn_x3_640_6gpu_foregroud-2(1-f)^2_backgroud-1_dynamic-10-2.yaml

```

```bash
python train_net.py --num-gpus 6 --config-file config_path
```
## 测试命令
```bash
python train_net.py --eval-only --num-gpus 6 --config-file config_path MODEL.WEIGHTS /path/to/checkpoint_file
```
## 测试命令2
```bash
测试FPS:
```

## tensorboard输出
```bash
tensorboard --logdir output/tensorboard
```

## ngrok内网穿透
```bash
ngrok http 6006
```

# 注意事项：
1. **修改文件，添加代码，必须要添加注释。**

2. **只允许新增配置文件，不允许在原有配置文件上更改，方便对比；修改后的配置文件，必须添加注释，说明修改内容。**

3. **注意修改不同文件的输出路径，避免文件覆盖。**

