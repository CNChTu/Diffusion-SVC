Language: [简体中文](./README.md) **English**

# Diffusion-SVC-Zero
[![Discord](https://img.shields.io/discord/1044927142900809739?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/jvA5c2xzSE)
***

<br>Samples and introductions can be found in [[Introduction Video(Not done yet)]]()

![Diagram](doc/diagram.jpg)
## 0. Introduction
Diffusion-SVC is a separate storage for the diffusion part of the [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) repository. It can be trained and inferred independently.

**The branch has replaced Diffusion-SVC with the same UNetConditionModel network as [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Add MRTE support for Zero Shot, Special thanks to [NS2VC](https://github.com/adelacvg/NS2VC)**

Disclaimer: Please ensure to only use **legally obtained authorized data** to train the Diffusion-SVC model, and do not use these models and any audio synthesized by them for illegal purposes. The author of this library is not responsible for any infringements, scams, and other illegal acts caused by the use of these model checkpoints and audio.

## 1. Install dependencies
1. Install PyTorch: We recommend first downloading PyTorch from the **[PyTorch official website](https://pytorch.org/)**
2. Install dependencies
```bash
pip install -r requirements.txt 
```

## 2. Configure the requirement model
- **(Required)** Download the pretrained [ContentVec](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/checkpoint_best_legacy_500.pt) encoder and place it in the `pretrain` folder.
  - Note: You can also use other feature extractors, but ContentVec is still highly recommended. All supported feature extractors can be found in the `Units_Encoder` class in `tools/tools.py`.
- **(Required)** Download the pretrained vocoder from the [DiffSinger community vocoder project](https://openvpi.github.io/vocoders) and unzip it to the `pretrain/` folder.
  -  Note: You should download the compressed file with `nsf_hifigan` in its name, not `nsf_hifigan_finetune`.

## 3. Preprocessing

### 1. Setting up Training and Validation Datasets

#### 1.1 Manual Configuration:

Place all training set data (.wav format audio clips) in the `data/train/audio` folder, or in a directory specified in the configuration file such as `xxxx/yyyy/audio`.

Place all validation set data (.wav format audio clips) in the `data/val/audio` folder, or in a directory specified in the configuration file such as `aaaa/bbbb/audio`.

#### 1.2 Random Selection by Program:

Run `python draw.py`. The program will help you select validation set data (you can adjust parameters such as the number of files to be extracted in `draw.py`).At this point, the audio needs to be placed in the `dataset_raw`

#### 1.3 Folder Structure Directory Display:
**Note: Speaker IDs must start from 1, not 0; if there is only one speaker, this speaker's ID must be 1.**
- Directory Structure:

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
#### 2. Start Preprocessing
```bash
python preprocess.py -c configs/config.yaml
```
You can modify the configuration file `configs/config.yaml` before preprocessing.

#### 3. Notes:
1. Please ensure that the sampling rate of all audio clips matches the sampling rate specified in the yaml configuration file! (If pre processing such as resampling is required, it is recommended to use [fap](https://github.com/fishaudio/audio-preprocess))

2. Cutting long audio into smaller clips can speed up training, but the duration of all audio clips should not be less than 6 seconds. If there are too many audio clips, you will need more memory. Setting the `cache_all_data` option in the configuration file to false can solve this problem.

3. It is suggested that the total number of audio clips in the validation set be about 10, don't put too many, otherwise the validation process will be slow.

4. If your dataset is not of high quality, set 'f0_extractor' to 'fcpe' in the configuration file.

5. The ‘n_spk’ parameter in the configuration file is invalid for this branch, just ignore it

## 4. Training

```bash
python train.py -c configs/config.yaml
```

## 5. Visualization
```bash
tensorboard --logdir=exp
```
After the first validation, you can see the synthesized test audio in TensorBoard.

## 6. Offline Inference
```bash
python main.py -i <input.wav> -ref <refer_audio.wav> -model <model_ckpt.pt> -o <output.wav> -k <keychange> -speedup <speedup> -method <method> -pe <f0_extractor> 
```
`-i` is the input source, `-ref` is the reference audio, `-model` is the path of the model, `-o` is the output audio  
`-k` is pitch change, `-speedup` is acceleration double speed, `-method` is `pndm`, `ddim`, `unipc`, `dpm-solver`  
`-pe` options are `crepe`, `parselmouth`, `dio`, `harvest`, `rmvpe`, `fcpe`, `fcpe` is recommended  

## Acknowledgement
* [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
* [soft-vc](https://github.com/bshall/soft-vc)
* [diff-SVC](https://github.com/prophesier/diff-SVC)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
* [NS2VC](https://github.com/adelacvg/NS2VC)
* [diffusers](https://github.com/huggingface/diffusers)

## Thank you to all contributors for their efforts
<a href="https://github.com/CNChTu/Diffusion-SVC/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNChTu/Diffusion-SVC" />
</a>
