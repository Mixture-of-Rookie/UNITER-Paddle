# Linux GPU/CPU 基础训练推理测试

Linux GPU/CPU 基础训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  UNITER  | uniter | 正常训练 | 正常训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  UNITER   |  uniter |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`lite_data`，直接使用即可。

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.1
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.1
    ```

- 安装依赖
    ```
    pip3 install  -r requirements.txt
    ```

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`uniter`的`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/prepare.sh test_tipc/configs/uniter/train_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/uniter/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - uniter - python3.6 tools/finetune_retrieval.py --cfg_file configs/retrieval_train_lite.yaml!
 Run successfully with command - uniter - python3.6 tools/export_model.py --cfg_file configs/retrieval_train_lite.yaml --out_dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0!
 Run successfully with command - uniter - python3.6 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/uniter/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !
 Run successfully with command - uniter - python3.6 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/uniter/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 ! 
 Run successfully with command - uniter - python3.6 -m paddle.distributed.launch --gpus=0,1,2,3 tools/finetune_retrieval.py --cfg_file configs/retrieval_train_lite.yaml! 
 Run successfully with command - uniter - python3.6 tools/export_model.py --cfg_file configs/retrieval_train_lite.yaml --out_dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0,1,2,3!
 Run successfully with command - uniter - python3.6 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0,1,2,3 --batch-size=1   --benchmark=True > ./log/uniter/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 ! 
 Run successfully with command - uniter - python3.6 deploy/inference_python/infer.py --use-gpu=False --model-dir=./log/uniter/lite_train_lite_infer/norm_train_gpus_0,1,2,3 --batch-size=1   --benchmark=True > ./log/uniter/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 ! 
```
