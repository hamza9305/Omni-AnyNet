#!/bin/bash

[ -z "$CONDA_ENV_NAME" ] && CONDA_ENV_NAME="omni-anynet"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME" || exit $?

[ -z "$INFERENCE_CHKPNT" ] && INFERENCE_CHKPNT="checkpoint/sceneflow/sceneflow.tar"
[ -z "$CUDA_DEVICE_ORDER" ] && export CUDA_DEVICE_ORDER="PCI_BUS_ID"
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0
[ -z "$DATASET" ] && DATASET="Dataset/theostereo_1024"
[ -z "$NUM_LOADER_WORKER" ] && NUM_LOADER_WORKER="20"
[ -z "$SPN_START" ] && SPN_START=5
[ -z "$BSIZE" ] && BSIZE=48
[ -z "$MASK_LUT_DIR" ] && MASK_LUT_DIR="masks_and_luts/"
[ -z "$MASK_FULL_RES" ] && MASK_FULL_RES="masks_and_luts/mask_full_res.pt"
[ -z "$MAX_DISP" ] && MAX_DISP=176

[ -z "$SAVE_PATH" ] && SAVE_PATH="inference_results_$(date --iso-8601='minutes' | sed 's/:/_/g')"

echo python inference.py --datapath "$DATASET" --save_path "$SAVE_PATH" \
        --with_spn --start_epoch_for_spn "$SPN_START" --test_bsize "$BSIZE" \
        --params "$INFERENCE_CHKPNT" --mask_lut_dir "$MASK_LUT_DIR" \
        --mask_new "$MASK_FULL_RES" --inference_set "test" --maxdisp "$MAX_DISP" \
        --baseline 0.3 --fov_deg 180 $@ \
        # --dump_euc_dist --dump_disp --dump_disp_index --dump_z_depth

date --iso-8601=ns
python inference.py --datapath "$DATASET" --save_path "$SAVE_PATH" \
        --with_spn --start_epoch_for_spn "$SPN_START" --test_bsize "$BSIZE" \
        --params "$INFERENCE_CHKPNT" --mask_lut_dir "$MASK_LUT_DIR" \
        --mask_new "$MASK_FULL_RES" --inference_set "test" --maxdisp "$MAX_DISP" \
        --baseline 0.3 --fov_deg 180 $@ \
        # --dump_euc_dist --dump_disp --dump_disp_index --dump_z_depth $@

date --iso-8601=ns
