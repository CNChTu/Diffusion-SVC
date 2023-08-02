python preprocess.py -c configs/config.yaml
python train.py -c configs/config.yaml
python train.py -c configs/config_naive.yaml
python combo.py -model C:\yourpath\exp\diffusion-test\model_2000.pt -nmodel C:\yourpath\exp\naive-test\model_10000.pt -exp output -n warioman
tensorboard --logdir=exp
python main.py -i input.wav -model C:\yourpath\exp\diffusion-test\model_yourepochs.pt -o output.wav -k 200 -id 0 -speedup 1 -method dpm-solver -nmodel C:\yourpath\exp\naive-test\model_yourepochs.pt
