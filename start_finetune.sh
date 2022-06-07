#!/bin/bash

[ -z "$CONDA_ENV_NAME" ] && CONDA_ENV_NAME="omni-anynet"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME" || exit $?

[ -z "$BETA" ] && BETA="2.0"
[ -z "$BASELINE" ] && BASELINE="0.3"
[ -z "$CHKPNT" ] && CHKPNT="checkpoint/sceneflow/sceneflow.tar"
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0
[ -z "$DATASET" ] && DATASET="Dataset/theostereo_1024"
[ -z "$EPOCHS" ] && EPOCHS=300
[ -z "$FOV_DEG" ] && FOV_DEG=180
[ -z "$LR" ] && LR=1e-3
[ -z "$NUM_LOADER_WORKER" ] && NUM_LOADER_WORKER="20"
[ -z "$SPN_START" ] && SPN_START=5
[ -z "$TEST_BSIZE" ] && TEST_BSIZE=48
[ -z "$TRAIN_BSIZE" ] && TRAIN_BSIZE=48
[ -z "$MASK_LUT_DIR" ] && MASK_LUT_DIR="masks_and_luts/"
[ -z "$MASK_FULL_RES" ] && MASK_FULL_RES="masks_and_luts/mask_full_res.pt"
[ -z "$MAX_DISP" ] && MAX_DISP=176


[ -z "$SAVE_PATH" ] && SAVE_PATH="train_results_BS_${TRAIN_BSIZE}_LR_${LR}_BETA_${BETA}_SPN_${SPN_START}_MAXEPOCHS_${EPOCHS}"
[ -z "$RESUME" ] && RESUME="${SAVE_PATH}/checkpoints"
[ -z "$FAILMAIL" ] && FAILMAIL="your.mail@here.com"

write_mail() {
    real_name=$(getent passwd $USER | cut -f5 -d: | cut -f1 -d,)
    from="$real_name <${USER}@${HOSTNAME}>"
    to="$FAILMAIL"
    subject="training failed"
    body="Training on ${HOSTNAME} failed. Training will be restarted in 1h."
    mail_msg="SUBJECT: ${subject}
TO: ${to}
CC: ${cc}
FROM: ${from}

$body"
    echo -e "$mail_msg" | sendmail "${to}"
}


while true
do
    # additional paramters: --with_tensorboard 
    nice python finetune.py --datapath "$DATASET" --dump_results --epochs "$EPOCHS"  --save_path "$SAVE_PATH" \
        --resume "$RESUME" --lr "$LR" --with_spn  --start_epoch_for_spn "$SPN_START"  --cosanneal \
        --train_bsize "$TRAIN_BSIZE" --test_bsize "$TEST_BSIZE" --plot_int 100 \
        --baseline "$BASELINE" --fov_deg "$FOV_DEG" --maxdisp "$MAX_DISP" \
        --beta "${BETA}" --pretrained "$CHKPNT" --mask_lut_dir "$MASK_LUT_DIR" --mask_new "$MASK_FULL_RES" $@
    exit_code=$?
    if [ "$exit_code" -eq "130" ];
    then
        echo "Training interrupted by user"
    elif [ "$exit_code" -eq "0" ];
    then
        echo "Training exited successfully."
        break
    else
        echo "Training stopped with exit code $exit_code ..."
        echo "Next try in 1 hour"
        # write_mail
        sleep "1h"
    fi
done

