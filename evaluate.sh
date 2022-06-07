#!/bin/bash

[ -z "$CONDA_ENV_NAME" ] && CONDA_ENV_NAME="omni-anynet"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME" || exit $?

[ -z "$BETA" ] && BETA="2.0"
[ -z "$CHKPNT" ] && CHKPNT="checkpoint/sceneflow/sceneflow.tar"
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0
[ -z "$DATASET" ] && DATASET="Dataset/theostereo_1024"
[ -z "$EPOCHS" ] && EPOCHS=300
[ -z "$LR" ] && LR=1e-3
[ -z "$NUM_LOADER_WORKER" ] && NUM_LOADER_WORKER="20"
[ -z "$SPN_START" ] && SPN_START=5
[ -z "$EVAL_BSIZE" ] && EVAL_BSIZE=48
[ -z "$MASK_LUT_DIR" ] && MASK_LUT_DIR="masks_and_luts/"
[ -z "$MASK_FULL_RES" ] && MASK_FULL_RES="masks_and_luts/mask_full_res.pt"
[ -z "$MAX_DISP" ] && MAX_DISP=176

[ -z "$SAVE_PATH" ] && SAVE_PATH="train_results_BS_${EVAL_BSIZE}_LR_${LR}_BETA_${BETA}_SPN_${SPN_START}_MAXEPOCHS_${EPOCHS}"
[ -z "$RESUME" ] && RESUME="${SAVE_PATH}/checkpoints"
[ -z "$FAILMAIL" ] && FAILMAIL="your.mail@here.com"

write_mail() {
    real_name=$(getent passwd $USER | cut -f5 -d: | cut -f1 -d,)
    from="$real_name <${USER}@${HOSTNAME}>"
    to="$FAILMAIL"
    subject="training failed"
    body="Evaluation on ${HOSTNAME} failed. Evaluation will be restarted in 1h."
    mail_msg="SUBJECT: ${subject}
TO: ${to}
CC: ${cc}
FROM: ${from}

$body"
    echo -e "$mail_msg" | sendmail "${to}"
}

[ -z "$FIRST_EPOCH" ] && FIRST_EPOCH="0"
[ -z "$LAST_EPOCH" ] && LAST_EPOCH=$((EPOCHS - 1))

if [ -n "$EVAL_SETS" ];
then
    IFS="," read -r -a EVAL_SETS <<< "$EVAL_SETS"
else
    EVAL_SETS=("train" "valid")
fi

echo "first epoch: $FIRST_EPOCH"
echo "last epoch:  $LAST_EPOCH"
echo "eval sets:"
last_set_id=${#EVAL_SETS[@]}
let last_set_id-=1
for i in $(seq 0 $last_set_id);
do
    echo "    ${EVAL_SETS[$i]}"
done

for e in $(seq $FIRST_EPOCH $LAST_EPOCH);
do
    for s in "${EVAL_SETS[@]}";
    do
        nice python finetune.py --datapath "$DATASET" --epochs "$EPOCHS" --save_path "$SAVE_PATH" \
            --load_epoch "$e" \
            --resume "$RESUME" --lr "$LR" --with_spn  --start_epoch_for_spn "$SPN_START"  --cosanneal \
            --train_bsize "$EVAL_BSIZE" --test_bsize "$EVAL_BSIZE" \
            --beta "${BETA}" --pretrained "$CHKPNT" --mask_lut_dir "$MASK_LUT_DIR" \
            --mask_new "$MASK_FULL_RES" --evaluate --evaluation_set "$s" --maxdisp "$MAX_DISP" \
            --baseline 0.3 --fov_deg 180 \
            $@
        exit_code=$?
        if [ "$exit_code" -eq "130" ];
        then
            echo "Evaluation interrupted by user"
            break
        elif [ "$exit_code" -eq "0" ];
        then
            echo "Evaluation on set \"$s\" exited successfully."
        else
            echo "Evaluation stopped with exit code $exit_code ..."
            # write_mail
            exit $exit_code
        fi
    done
done

