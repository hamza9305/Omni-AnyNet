#!/bin/bash

function abortOnFailure {
    status=$?
    if [ $status -ne "0" ];
    then
        echo "ERROR CODE: $status"
        exit $status
    fi
}


TITLE="AnyNet Evaluation"
MENU_HEIGHT=30
MENU_WIDTH=80
MENU_OPTIONS=20

cur_branch=$(git branch --show-current)
if (whiptail --title "${TITLE}" --yesno "You are going to evaluate on the branch ${cur_branch}. Is that correct?" ${MENU_HEIGHT} ${MENU_WIDTH});
then
    echo "Evaluating on the branch ${cur_branch}"
else
    exit 0
fi

parameters=$(whiptail --title "${TITLE}" --checklist \
"Parameters:" ${MENU_HEIGHT} ${MENU_WIDTH} ${MENU_OPTIONS} \
"dump_error_maps" "Dump error maps on disc for offline evaluation" ON \
"with_tensorboard" "Write results to tensorboard" OFF \
3>&1 1>&2 2>&3)
abortOnFailure
parameters=${parameters//\"/}
parameters=$(echo "$parameters" | sed -E "s/([^[:space:]]*)/--\1/g")
parameters=$(echo "$parameters" | xargs)

export EVAL_SETS="test"
EVAL_GRAPH_NAME="eval_${EVAL_SETS}"
case "${parameters[@]}" in
*"--with_tensorboard"*)
    EVAL_GRAPH_NAME=$(whiptail --title "${TITLE}" --inputbox "Evaluation graph name:" ${MENU_HEIGHT} ${MENU_WIDTH} "$EVAL_GRAPH_NAME" 3>&1 1>&2 2>&3);
    abortOnFailure
    EVAL_GRAPH_NAME=${EVAL_GRAPH_NAME//\"/}
    if [ -n "$EVAL_GRAPH_NAME" ];
    then
        parameters+=('--eval_graph_name')
        parameters+=("$EVAL_GRAPH_NAME")
    fi
    ;;
esac

DEF_EPOCH=295
EPOCH=$(whiptail --title "${TITLE}" --inputbox "Epoch to evaluate:" ${MENU_HEIGHT} ${MENU_WIDTH} "$DEF_EPOCH" 3>&1 1>&2 2>&3);
abortOnFailure
EPOCH=${EPOCH//\"/}
export FIRST_EPOCH="$EPOCH"
export LAST_EPOCH="$EPOCH"

bash evaluate.sh ${parameters[@]}
