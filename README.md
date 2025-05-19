# ST-MEM: Spatio-Temporal Masked Electrocardiogram Modeling

This is an official implementation of "Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram".

Paper: https://openreview.net/pdf?id=WcOohbsF4H

## Environments
### Requirements
- python 3.9
- pytorch 1.11.0
- einops 0.6.0
- mergedeep 1.3.4
- numpy 1.21.6
- pandas 1.4.2
- PyYAML 6.0
- scipy 1.8.1
- tensorboard
- torchmetrics
- tqdm
- wfdb

### Installation
```console
(base) user@server:~$ conda create -n st_mem python=3.9
(base) user@server:~$ conda activate st_mem
(st_mem) user@server:~$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
(st_mem) user@server:~$ git clone https://github.com/bakqui/ST-MEM.git
(st_mem) user@server:~$ cd ST-MEM
(st_mem) user@server:~/ST-MEM$ pip install -r requirements.txt
```

## Pre-training

To pre-train ST-MEM with ViT-B/75 encoder, run the following:
```
bash run_pretrain.sh \
    --gpus ${GPU_IDS} \
    --config_path ./configs/pretrain/st_mem.yaml \
    --output_dir ${OUTPUT_DIRECTORY} \
    --exp_name ${EXPERIMENT_NAME}
```

We present the pre-trained ST-MEM:
- Encoder: https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view?usp=share_link
- Full (encoder + decoder): https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view?usp=share_link

## Downstream training

To fine-tune the ST-MEM ViT-B/75 encoder, run the following:
```
bash run_downstream.sh \
    --gpus ${GPU_IDS} \
    --config_path ./configs/downstream/st_mem.yaml \
    --output_dir ${OUTPUT_DIRECTORY} \
    --exp_name ${EXPERIMENT_NAME} \
    --encoder_path ${PRETRAINED_ENCODER_PATH}
```

## License
© VUNO Inc. All rights reserved.
 
This repository contains code developed at VUNO Inc. by its employees as part of their official duties.
Do not distribute, modify, or use this code outside the scope permitted by the license without explicit permission from VUNO.

## Citation

If you find this work or code is helpful in your research, please cite:
```
@inproceedings{na2024guiding,
  title     = {Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram},
  author    = {Na, Yeongyeon and 
               Park, Minje and 
               Tae, Yunwon and 
               Joo, Sunghoon},
  booktitle = {International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=WcOohbsF4H}
}
```
