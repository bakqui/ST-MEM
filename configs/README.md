# Training Configurations

We prepared the basic configurations of pre-training and downstream fine-tuning of MTAE, MLAE, and ST-MEM.


## Init config
- `seed`: random seed.
- `output_dir`: base directory where the training outcomes (e.g., model checkpoints, logs, and metric summary) are saved
- `exp_name`: experimental directory name
- `resume`: resume checkpoint path
- `start_epoch`: resume start epoch
- `encoder_path`: pre-trained encoder checkpoint path.
- `model_name`: model init function name declared in `./models` and `./models/encoder` for pre-training and downstream training, respectively
- `mode`: `finetune` or `linprobe` or `scratch`, only for downstream training.

## `model`
- `seq_len`: pre-defined ECG input length (sampling rate (Hz) * time (sec))
- `patch_size`: ViT patch size
- `num_leads`: number of leads (channels)
- `num_classes`: number of classes in multi-class classification task

## `dataset`
|       | **FILE_NAME** | **SAMPLE_RATE** | **LABEL** |
|:-----:|:-------------:|:---------------:|:---------:|
| **0** |   00001.pkl   |       500       |     0     |
| **1** |   00002.pkl   |       500       |     1     |
| **2** |   00003.pkl   |       250       |     0     |
- `filename_col`: ECG file name column of index file (e.g., `FILE_NAME`)
- `fs_col`: ECG sampling rate column name of index file (e.g., `SAMPLE_RATE`)
- `label_col`: ECG label column name of index file (e.g., `LABEL`)
- `fs`: resampling rate to make all the ECGs to be same length
- `index_dir`: directory where the index file is saved
- `ecg_dir`: directory where the ECG signal files are saved
- `{train / valid / test}_csv`: training / validation / test index file name
- `rand_augment` / `train_transforms` / `eval_transforms`: ECG transform configurations for preprocessing and augmentation

## `dataloader`
Please refer to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

## `train`
- `epochs`: number of training epochs
- `accum_iter`: number of gradient accumulation steps (_effective batch size_ = _number of DDP processes_ * `batch_size` * `accum_iter`)
- `warmup_epochs`: number of epochs for learning rate warm-up
- `min_lr`: minimum learning rate
- `blr`: base learning rate
- `lr`: actual learning rate (It is computed by the linear scaling rule unless it is explicitly given: `lr` = `blr` * _effective batch size_ / 256)
- `weight_decay`: weight decay coefficient
- `max_norm`: maximum norm of gradients for gradient clipping
- `optimizer`: `adamw` or `sgd`

## `metric`
Please refer to https://lightning.ai/docs/torchmetrics/stable/all-metrics.html

