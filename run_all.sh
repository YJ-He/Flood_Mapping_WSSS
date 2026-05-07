#!/bin/bash
# Run training + testing for all three flood datasets sequentially.
# Usage:
#   bash run_all.sh         # default GPU 0
#   bash run_all.sh 1       # use GPU 1

set -e

GPU=${1:-0}
CONFIGS=("configs/config_UAVflood.cfg" "configs/config_SARflood.cfg" "configs/config_GFflood.cfg")
NAMES=("UAVflood(3ch)" "SARflood(8ch)" "GFflood(5ch)")

TOTAL_START=$SECONDS

echo "======================================"
echo " Flood Mapping - Train & Test All"
echo " GPU: ${GPU}"
echo "======================================"

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    START=$SECONDS

    echo ""
    echo "======================================"
    echo " [$((i+1))/3] ${NAME}"
    echo " Config: ${CFG}"
    echo "======================================"

    python train_flood.py -config "${CFG}" -gpu "${GPU}"

    ELAPSED=$(( SECONDS - START ))
    echo ""
    echo " [${NAME}] finished in $((ELAPSED/60))m $((ELAPSED%60))s"
done

TOTAL=$(( SECONDS - TOTAL_START ))
echo ""
echo "======================================"
echo " All done! Total time: $((TOTAL/60))m $((TOTAL%60))s"
echo "======================================"
