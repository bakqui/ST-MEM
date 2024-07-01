#!/bin/bash

# help
function usage()
{
    cat <<EOM
Usage: bash $0 [options]
Options:
  --master_port PORT              Master port (default=12345)
  --gpus GPUS                     GPU indices (default=0)
  -f, --config_path PATH          Path of config file (required)
  --output_dir PATH               Output directory (optional)
  --exp_name NAME                 Experiment name (optional)
  --resume PATH                   Path of checkpoint to resume (optional)
  --start_epoch EPOCH             Start epoch (optional)
  -h, --help                      Print help
EOM

    exit 1
}


# parser
function set_options()
{
    arguments=$(getopt --options f:h \
                       --longoptions master_port:,gpus:,config_path:,output_dir:,exp_name:,resume:,start_epoch:,help \
                       --name $(basename $0) \
                       -- "$@")

    eval set -- "$arguments"

    while true
    do
        case "$1" in
            --master_port)
                MASTER_PORT=$2
                shift 2
                ;;
            --gpus)
                GPUS=$2
                shift 2
                ;;
            -f | --config_path)
                CONFIG_PATH=$2
                shift 2
                ;;
            --output_dir)
                OUTPUT_DIR=$2
                shift 2
                ;;
            --exp_name)
                EXP_NAME=$2
                shift 2
                ;;
            --resume)
                RESUME=$2
                shift 2
                ;;
            --start_epoch)
                START_EPOCH=$2
                shift 2
                ;;
            --)
                shift
                break
                ;;
            -h | --help)
                usage
                ;;
        esac
    done
}

# default
MASTER_PORT=12345
GPUS="0"

# parsing
set_options "$@"

# check required arguments
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: config_path is required"
    usage
fi

# print parsed arguments
echo "Arguments"
echo -e "\tMASTER_PORT: ${MASTER_PORT}"
echo -e "\tGPUS: ${GPUS}"
echo -e "\tCONFIG_PATH: ${CONFIG_PATH}"
echo -e "\tOUTPUT_DIR: ${OUTPUT_DIR}"
echo -e "\tEXP_NAME: ${EXP_NAME}"
echo -e "\tRESUME: ${RESUME}"
echo -e "\tSTART_EPOCH: ${START_EPOCH}"

# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPUS

# run downstream training
NUM_GPUS=$(echo $GPUS | tr "," "\n" | wc -l)

TORCHRUN_ARGS="--master_port $MASTER_PORT \
               --nproc_per_node $NUM_GPUS"
MAIN_ARGS="--config_path $CONFIG_PATH"

if [ ! -z ${OUTPUT_DIR} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --output_dir $OUTPUT_DIR"
fi
if [ ! -z ${EXP_NAME} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --exp_name $EXP_NAME"
fi
if [ ! -z ${RESUME} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --resume $RESUME"
fi
if [ ! -z ${START_EPOCH} ]; then
    MAIN_ARGS="$MAIN_ARGS \
               --start_epoch $START_EPOCH"
fi

echo "Run main..."
if [ $NUM_GPUS -gt 1 ]; then
    torchrun $TORCHRUN_ARGS main_pretrain.py $MAIN_ARGS
else
    python main_pretrain.py $MAIN_ARGS
fi
