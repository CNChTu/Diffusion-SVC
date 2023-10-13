# Diffusion-SVC-Zero

## *. Warning (Please read carefully ! ! !)
0. The development of any one-click packages is strictly prohibited ! ! !
1. Please resolve dataset licensing issues on your own and refrain from using unauthorized datasets for training. Any problems arising from the use of unauthorized datasets for training are your sole responsibility, and you will bear all the associated consequences. This is unrelated to the repository or its maintainers.
2. This project is established for academic exchange purposes, intended solely for communication and learning, and is not prepared for production environments.
3. You are solely responsible for any copyright infringement issues caused by the input sources, and you will bear all associated consequences. When using other commercial singing synthesis software as input sources, please ensure compliance with the software's terms of use. Note that many singing synthesis engines explicitly state that they cannot be used as input sources for conversion!
4. The use of this project for illegal activities, as well as religious and political purposes, is strictly prohibited. The project maintainers strongly oppose such activities. If you do not agree with this policy, you are not allowed to use this project.
5. Continued use of this repository is considered as an agreement to abide by the terms and conditions outlined in the repository's README. The README has been provided for guidance, and the repository is not responsible for any potential issues that may arise in the future.
6. If you intend to use this project for any other initiatives or purposes, please contact and inform the repository's author in advance. Your cooperation is greatly appreciated.

## 0. Introduction
Diffusion-SVC is a separate storage for the diffusion part of the [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) repository. It can be trained and inferred independently.

**This branch has replaced Diffusion-SVC with the same UNetConditionModel network as [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Add MRTE support for Zero Shot, Special thanks to [NS2VC](https://github.com/adelacvg/NS2VC)**

## 1. Install dependencies
1. Install PyTorch based on your operating system and hardware. **[PyTorch official website](https://pytorch.org/)**
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

### 1. Notes

1. It is recommended that the total duration of the dataset is not less than 1000 hours, with a minimum of 2000 speakers.
2. Regardless of whether it's the original audio or concatenated audio, ensure that each audio clip has only one timbre.
3. Please ensure that the sampling rate of all audio clips matches the sampling rate specified in the yaml configuration file! (If pre processing such as resampling is required, it is recommended to use [fap](https://github.com/fishaudio/audio-preprocess))
4. Cutting long audio into smaller clips can save VRAM, but the duration of all audio clips should not be less than 6 seconds. Audio clips with a duration of less than 6 seconds need to be concatenated to exceed 6 seconds.
5. It is suggested that the total number of audio clips in the validation set be about 10, don't put too many, otherwise the validation process will be very slow.
6. If your dataset is not of high quality, set 'f0_extractor' to 'fcpe' in the configuration file.
7. The ‘n_spk’ parameter in the configuration file is invalid for this branch, just ignore it

### 2. Setting up Training and Validation Datasets

#### 1.1 Manual Configuration

Place all training set data (.wav format audio clips) in the `data/train/audio` folder, or in a directory specified in the configuration file such as `xxxx/yyyy/audio`.

Place all validation set data (.wav format audio clips) in the `data/val/audio` folder, or in a directory specified in the configuration file such as `aaaa/bbbb/audio`.

#### 1.2 Random Selection by Program

Run `python draw.py`. The program will help you select validation set data (you can adjust parameters such as the number of files to be extracted in `draw.py`).At this point, the audio needs to be placed in the `dataset_raw`

#### 1.3 Folder Structure Directory Display

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
### 3. Start Preprocessing

Modify the configuration file based on the comments within the reference file `configs/config.yaml`.

Then, start the preprocessing
```bash
python preprocess.py -c configs/config.yaml
```

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
