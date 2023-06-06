Language: [简体中文](./README.md) **English**

**This English readme will not update first time, please read [简体中文](./README.md) for latest info。**

I am not good at English. If there are any errors, please point them out.

# Diffusion-SVC
This repository is a separate storage for the diffusion part of the [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) repository. It can be trained and inferred independently.

## 0. Introduction
Diffusion-SVC is a separate storage for the diffusion part of the [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) repository. It can be trained and inferred independently.

Compared with the well-known [Diff-SVC](https://github.com/prophesier/diff-svc), this project consumes much less graphic memory, has faster training and inference speed, and is specially optimized for shallow diffusion and real-time use. It can perform real-time inference on a powerful GPU.

If the quality of the training data and the input source are both very high, Diffusion-SVC may have the best conversion effect.

In addition, this project can easily be cascaded after other acoustic models for shallow diffusion, to improve the final output effect or reduce performance consumption. For example, cascading Diffusion-SVC after [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) can further reduce the number of diffusion steps required and obtain high-quality output.

Disclaimer: Please ensure to only use **legally obtained authorized data** to train the Diffusion-SVC model, and do not use these models and any audio synthesized by them for illegal purposes. The author of this library is not responsible for any infringements, scams, and other illegal acts caused by the use of these model checkpoints and audio.

## 1. Install dependencies
1. Install PyTorch: We recommend first downloading PyTorch from the **[PyTorch official website](https://pytorch.org/)**, then run:
```bash
pip install -r requirements.txt 
```

## 2. Configure the requirement model
- **(Required)** Download the pretrained [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) encoder and place it in the `pretrain` folder.
  - Note: You can also use other feature extractors, but ContentVec is still highly recommended. All supported feature extractors can be found in the `Units_Encoder` class in `tools/tools.py`.
- **(Required)** Download the pretrained vocoder from the [DiffSinger community vocoder project](https://openvpi.github.io/vocoders) and unzip it to the `pretrain/` folder.
  -  Note: You should download the compressed file with `nsf_hifigan` in its name, not `nsf_hifigan_finetune`.
- If you need to use the voiceprint model, you need to set `use_speaker_encoder` in the configuration file to `true`, and download the pretrained voiceprint model from [here](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3?usp=sharing). This model comes from [mozilla/TTS](https://github.com/mozilla/TTS/wiki/Released-Models).

## 3. Preprocessing

### 1. Setting up Training and Validation Datasets

#### 1.1 Manual Configuration:

Place all training set data (.wav format audio clips) in the `data/train/audio` folder, or in a directory specified in the configuration file such as `xxxx/yyyy/audio`.

Place all validation set data (.wav format audio clips) in the `data/val/audio` folder, or in a directory specified in the configuration file such as `aaaa/bbbb/audio`.

#### ~~1.2 Random Selection by Program (Not Implemented):~~

~~Run `python draw.py`. The program will help you select validation set data (you can adjust parameters such as the number of files to be extracted in `draw.py`).~~

#### 1.3 Folder Structure Directory Display:
**Note: Speaker IDs must start from 1, not 0; if there is only one speaker, this speaker's ID must be 1.**
- Directory Structure:

```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ccc.wav
│    │    │   ├─ ddd.wav
│    │    │   └─ ....wav
│    │    └─ ...
|
├─ val
|    ├─ audio
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ggg.wav
│    │    │   ├─ hhh.wav
│    │    │   └─ ....wav
│    │    └─ ...
```

#### 2. Start Preprocessing
```bash
python preprocess.py -c configs/config.yaml
```
You can modify the configuration file `configs/config.yaml` before preprocessing.

#### 3. Notes:
1. Please ensure that the sampling rate of all audio clips matches the sampling rate specified in the yaml configuration file! If they don't match, the program can still run, but resampling during training will be very slow. (Optional: You can use Adobe Audition™'s Match Loudness function to resample, modify channels, and match loudness all at once.)

2. Cutting long audio into smaller clips can speed up training, but the duration of all audio clips should not be less than 2 seconds. If there are too many audio clips, you will need more memory. Setting the `cache_all_data` option in the configuration file to false can solve this problem.

3. It is suggested that the total number of audio clips in the validation set be about 10, don't put too many, otherwise the validation process will be slow.

4. If your dataset quality is not very high, please set the 'f0_extractor' in the configuration file to 'crepe'. The crepe algorithm has the best noise resistance, but it will significantly increase the time required for data preprocessing.

5. The ‘n_spk’ parameter in the configuration file will control whether to train a multi-speaker model. If you want to train a **multi-speaker** model, to number the speakers, all audio folder names must be **integers no larger than ‘n_spk’**.

## 4. Training

### 1. Training without pre-trained model:
```bash
python train.py -c configs/config.yaml
```

### 2. Pre-trained Models:
**We strongly recommend fine-tuning with pre-trained models, which is easier and much more time-saving than training from scratch, and can achieve a higher limit than small datasets.**

**Please note, fine-tuning on a base model requires using the same encoder as the base model, such as ContentVec, the same applies to other encoders (like voiceprint), and the model's network size and other parameters should be the same.**

| Encoder Used                                                                                                                                                                                | Network Size | Dataset           | Download                                                                                                          |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|-------------------|-------------------------------------------------------------------------------------------------------------------|
| [contentvec768l12(Recommended)](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)                                                                                                 | 512*20       | VCTK<br/>m4singer | [HuggingFace](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/contentvec768l12.7z)                   |
| [contentvec768l12](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)<br/>+[use_spk_encoder](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3?usp=sharing) | 512*20       | VCTK<br/>m4singer | [HuggingFace](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/contentvec768l12%2Buse_spk_encoder.7z) |
| [hubertsoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)                                                                                               | 512*20       | VCTK<br/>m4singer | [HuggingFace](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/hubertsoft.7z)                         |
| [wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)                                                                                              | 512*20       | VCTK<br/>m4singer | [HuggingFace](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/wav2vec2ctc.7z)                        |

Here is an additional special pre-trained model using the contentvec768l12 encoder, the dataset is `m4singer`/`opencpop`/`vctk`. It is not recommended to use this and there's no guarantee it won't cause problems: [Download](https://huggingface.co/ChiTu/Diffusion-SVC/resolve/main/v0.1/contentvec768l12%2Bmakefunny.7z).

### 3. Training with Pretrained Models:
1. We welcome pull requests for multi-speaker pretrained models (please use datasets that are authorized for open-source training).
2. The pretrained models are mentioned above. Please note that the model must use the same encoder.
3. Place the pretrained model named `model_0.pt` in the model export folder specified by the "expdir: exp/*****" parameter in `config.yaml`. If the folder doesn't exist, create a new one. The program will automatically load the pretrained model from this folder.
4. Start training just like training without pretrained model.

## 5. Visualization
```bash
# Use tensorboard to check the training status
tensorboard --logdir=exp
```
After the first validation, you can see the synthesized test audio in TensorBoard.

## 6. Offline Inference
```bash
python main.py -i <input.wav> -model <model_ckpt.pt> -o <output.wav> -k <keychange> -id <speaker_id> -speedup <speedup> -method <method> -kstep <kstep>
```
`-model` is the model path, `-k` is the pitch shift, `-speedup` is the speedup multiplier, `-method` is either `pndm` or `dpm-solver`, `-kstep` is the shallow diffusion step, `-id` is the speaker ID of the diffusion model.

If `-kstep` is not empty, shallow diffusion will be performed on the input source mel, if `-kstep` is empty, full depth Gaussian diffusion will be performed.

If voiceprint encoding was used, an external voiceprint can be specified with `-spkemb`, or the model's voiceprint dictionary can be overwritten with `-spkembdict`.

## 7. Units Index(Optional,Not Recommended)
Like [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc).

**Note that this is an optional feature and can be inferred normally without use.Indexing takes up a large amount of storage space and also consumes a lot of CPU during indexin.**
```bash
# training index，preprocessing needs to be completed first
python train_units_index.py -c config.yaml
```
推理时，使用`-lr`参数使用。此参数为检索比率。

## 8. Real-Time Inference

This project can work with [rtvc](https://github.com/fishaudio/realtime-vc-gui) to achieve real-time inference.

**Note: This is an experimental feature at present, rtvc is also not fully developed, so it's not recommended for use.**

```bash
# Needs to be used in conjunction with rtvc
python flask_api.py
```

## 9. Compatibility
### 9.1. Units Encoder
|                               | Diffusion-SVC | [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) | [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) |
|-------------------------------|---------------|------------------------------------------------|----------------------------------------------------------------|
| ContentVec                    | √             | √                                              | √                                                              |
| HubertSoft                    | √             | √                                              | √                                                              |
| Hubert(Base,Large)            | √             | √                                              | ×                                                              |
| CNHubert(Base,Large)          | √             | √                                              | √*                                                             |
| CNHubertSoft                  | √             | √                                              | ×                                                              |
| Wav2Vec2-xlsr-53-espeak-cv-ft | √*            | ×                                              | ×                                                              |
| DPHubert                      | ×             | ×                                              | √                                                              |
| Whisper-PPG                   | ×             | ×                                              | √*                                                             |


## Acknowledgement
* [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
* [soft-vc](https://github.com/bshall/soft-vc)
* [diff-SVC](https://github.com/prophesier/diff-SVC)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
