Language: [English](./README_en.md) **简体中文**

# Diffusion-SVC-Zero
[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/jvA5c2xzSE)

***

<br>效果和介绍见[[介绍视频(暂未完成)]]()
**欢迎加群交流讨论：882426004**
![Diagram](doc/diagram.jpg)
## 0.简介
**Diffusion-SVC 是[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)仓库的扩散部分的单独存放。可单独训练和推理。**

**本分支的Diffusion-SVC更换为了[Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui)的同款网络UNetConditionModel，增加MRTE支持Zero-Shot，特别鸣谢[NS2VC](https://github.com/adelacvg/NS2VC)**

免责声明：请确保仅使用**合法获得的授权数据**训练 Diffusion-SVC 模型，不要将这些模型及其合成的任何音频用于非法目的。 本库作者不对因使用这些模型检查点和音频而造成的任何侵权，诈骗等违法行为负责。

## 1. 安装依赖
1. 安装PyTorch：我们推荐从 **[PyTorch 官方网站 ](https://pytorch.org/)** 下载 PyTorch.

2. 安装依赖
```bash
pip install -r requirements.txt 
```

## 2. 配置预训练模型
- **(必要操作)** 下载预训练 [ContentVec](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/checkpoint_best_legacy_500.pt) 编码器并将其放到 `pretrain` 文件夹。
  - 注意：也可以使用别的特征提取，但仍然优先推荐ContentVec。支持的所有特征提取见`tools/tools.py`中的`Units_Encoder`类。
- **(必要操作)** 从 [DiffSinger 社区声码器项目](https://openvpi.github.io/vocoders) 下载预训练声码器，并解压至 `pretrain/` 文件夹。
  -  注意：你应当下载名称中带有`nsf_hifigan`的压缩文件，而非`nsf_hifigan_finetune`。

## 3. 预处理

### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`,也可以是配置文件中指定的文件夹如`xxxx/yyyy/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`,也可以是配置文件中指定的文件夹如`aaaa/bbbb/audio`。

#### 1.2 程序随机选择：

运行`python draw.py`,程序将帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）。此时音频要放到`dataset_raw`中

#### 1.3文件夹结构目录展示：
**注意：说话人id必须从1开始，不能从0开始；如果只有一个说话人则该说话人id必须为1**
- 目录结构：

```
data
├─ train
│    └─ audio
│         ├─ aaa.wav
│         ├─ bbb.wav
│         └─ ....wav
│         
|
├─ val
|    └─ audio
│         ├─ eee.wav
│         ├─ fff.wav
│         └─ ....wav
│         
└─ 
```

### 2. 正式预处理
```bash
python preprocess.py -c configs/config.yaml
```
您可以在预处理之前修改配置文件 `configs/config.yaml`

### 3. 备注：
1. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！（推荐使用[fap](https://github.com/fishaudio/audio-preprocess)进行重采样的等前处理）

2. 长音频切成小段可以加快训练速度，但所有音频切片的时长不应少于 6 秒。如果音频切片太多，则需要较大的内存，配置文件中将 `cache_all_data` 选项设置为 false 可以解决此问题。

3. 验证集的音频切片总数建议为 10 个左右，不要放太多，不然验证过程会很慢。

4. 如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'fcpe'。

5. 配置文件中的 ‘n_spk’ 参数对于此分支无效，无视即可

## 4. 训练

```bash
python train.py -c configs/config.yaml
```

## 5. 可视化
```bash
tensorboard --logdir=exp
```
第一次验证后，在 TensorBoard 中可以看到合成后的测试音频。

## 6. 非实时推理
```bash
python main.py -i <input.wav> -ref <refer_audio.wav> -model <model_ckpt.pt> -o <output.wav> -k <keychange> -speedup <speedup> -method <method> -pe <f0_extractor> 
```
`-i`是输入源，`-ref`是参考音频，`-model`是模型的路径，`-o`是输出的音频  
`-k`是变调， `-speedup`为加速倍速，`-method`为`pndm`，`ddim`，`unipc`，`dpm-solver`  
`-pe` 可选项为 `crepe`，`parselmouth`，`dio`，`harvest`，`rmvpe`，`fcpe`，推荐 `fcpe`  

## 感谢
* [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
* [soft-vc](https://github.com/bshall/soft-vc)
* [diff-SVC](https://github.com/prophesier/diff-SVC)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
* [NS2VC](https://github.com/adelacvg/NS2VC)
* [diffusers](https://github.com/huggingface/diffusers)
## 感谢所有贡献者作出的努力
<a href="https://github.com/CNChTu/Diffusion-SVC/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNChTu/Diffusion-SVC" />
</a>
