# UNITER-Paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/abs/1909.11740)实现

## 一、简介

本项目使用[paddle](https://github.com/PaddlePaddle/Paddle)框架复现[UNITER](https://arxiv.org/abs/1909.11740)模型。该模型借助目标类别`Object Tags`来实现更好的视觉和文本的跨模态对齐。作者引入`Object Tags`并基于此提出了两个损失函数进行大规模的预训练，使得能够学习到文本和图像区域的语义对齐表征。实验表明，作者在多个 vision-language 任务上得到了有效的提升。

**注:**

**AI Studio多卡项目地址: [https://aistudio.baidu.com/aistudio/clusterprojectdetail/3496916](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3496916).**

**AI Studio单卡项目地址: [https://aistudio.baidu.com/aistudio/projectdetail/3496946](https://aistudio.baidu.com/aistudio/projectdetail/3496946).**

**您可以使用[AI Studio](https://aistudio.baidu.com/)平台在线运行该项目!**

**论文:**

* [1] Y. Chen, L. Li, L. Yu, and et. al, "UNITER: UNiversal Image-TExt Representation Learning", ECCV, 2020.

**参考项目:**

* [UNITER](https://github.com/ChenRocks/UNITER) [官方实现]

## 二、复现精度

> 本项目验证其在图文检索`Image-Text Retrieval`下游任务中的性能，所使用的数据集为[Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)，复现精度如下。

| 指标 | 原论文 | 复现精度 |
| :---: | :---: | :---: |
| IR-flickr30K-R1 | 73.66 | 74.02 |


## 三、数据集

本项目所使用的数据集为[Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)。该数据集共包含31783张图像，每张图像对应5个标题。训练集、验证集和测试集分别为29783、1000、1000张图像及其对应的标题。本项目使用预提取的`bottom-up`特征，可以从[这里](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_itm.sh)下载得到。


## 四、环境依赖

* 硬件：CPU、GPU

* 软件：
    * Python 3.7
    * PaddlePaddle-GPU == 2.2.1
    * PaddleNLP==2.2.1

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/Mixture-of-Rookie/UNITER-Paddle.git
cd UNITER-Paddle
```

### step2: 安装环境及依赖

```bash
pip install -r requirements.txt
```

### step3: 挂载数据

```bash
# 相关数据集已上传至Aistudio
# 详情见: https://aistudio.baidu.com/aistudio/datasetdetail/128538

# paddle格式的预训练权重也已上传至Aistudio
# 详情见: https://aistudio.baidu.com/aistudio/datasetdetail/128538

# 下载或挂载数据集和预训练权重之后
# 需要修改配置文件(configs/retrieval_train.yaml和configs/retrieval_test.yaml的一些参数:
# DATA_DIR (数据集目录), FEAT_FILE (特征文件), PRETRAINED-DIR (预训练权重路径)
```

### step4: 训练

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES='0, 1, 2, 3' python -m paddle.distributed.launch tools/finetune_retrieval.py --cfg_file configs/retrieval_train.yaml
```

### step5: 测试

```bash
# 测试之前,需要在configs/retrieval_test.yaml中指定测试的模型 (即修改EVAL-CHECKPOINT_DIR参数).
python tools/evaluate_retrieval.py --cfg_file configs/retrieval_test.yaml
```

### 使用预训练模型进行预测

```bash
# 下载训练好的模型权重
# https://aistudio.baidu.com/aistudio/datasetdetail/129717
# 执行Step5进行测试
```

### 模型动转静导出

```
python tools/export_model.py --cfg_file configs/retrieval_train_lite.yaml
```

最终在`./static/`文件夹下会生成下面的3个文件：

```
inference_output
  |----model.pdiparams     : 模型参数文件
  |----model.pdmodel       : 模型结构文件
  |----model.pdiparams.info: 模型参数信息文件
```

### 模型推理

```
python deploy/inference_python/infer.py
```

## 六、TIPC

```bash
# 准备数据和环境
bash test_tipc/prepare.sh test_tipc/configs/uniter/train_infer_python.txt lite_train_lite_infer
# test tipc
bash test_tipc/test_train_inference_python.sh test_tipc/configs/uniter/train_infer_python.txt lite_train_lite_infer
```

> Tips:
>
> *  TIPC需安装[AutoLog](https://github.com/LDOUBLEV/AutoLog)。
>
> * TIPC需使用develop分支的paddlepaddle，详情见[issue](https://github.com/PaddlePaddle/Paddle/issues/43225#issuecomment-1154764339)。

## 七、代码结构与详细说明

```bash
├── config                    # 默认配置文件夹
│   └── default.py            # 默认配置参数
├── configs                   # 指定配置文件夹
│   └── retrieval_train.yaml  # 训练配置文件
│   └── retrieval_test.yaml   # 测试配置文件
├── datasets
│   └── retrieval_dataset.py  # 数据加载
├── models
│   └── bert.py               # bert模型
│   └── uniter.py             # uniter模型
│   └── uniter_retrieval.py   # uniter模型
├── solvers
│   └── optimizer.py          # 优化器
│   └── scheduler.py          # 学习率策略
├── tools
│   └── finetune_retrieval.py # 训练脚本
│   └── evaluate_retrieval.py # 测试脚本
└── requirement.txt           # 依赖包
```

## 八、模型信息

关于模型的其他信息，可以参考下表：

|   信息   |                             说明                             |
| :------: | :----------------------------------------------------------: |
|  发布者  |                           fuqianya                           |
|   时间   |                           2022.05                            |
| 框架版本 |                         Paddle 2.2.1                         |
| 应用场景 |                            多模态                            |
| 支持硬件 |                           GPU、CPU                           |
| 下载链接 | [预训练模型](https://aistudio.baidu.com/aistudio/datasetdetail/129717) \| [训练日志](https://aistudio.baidu.com/aistudio/datasetdetail/129717) |
