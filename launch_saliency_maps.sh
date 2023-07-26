# Parameters
ROOT="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/root"
MODEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/models"
SAVING_DIR="${MODEL_DIR}/20230111_schizophrenic_wo_ventricle_skeletons/sleek-sweep-2"
PREPROC="skeleton" # "vbm", "quasi_raw" or "skeleton"
PB="scz" # "scz",  "asd" or "bipolar"
SALIENCY_METHOD="gradient"
NET="densenet121" #"densenet121",  "resnet18" or "alexnet"
EXP_NAME="${PB}_${PREPROC}_${NET}"
EPOCH=99

FOLD=0

CHECKPOINT="${SAVING_DIR}/${EXP_NAME}_${FOLD}_epoch_${EPOCH}.pth"

# Command
python3 sml_training/run_saliency_maps.py --root $ROOT  \
--saving_dir $SAVING_DIR --preproc $PREPROC \
--pb $PB --saliency_meth $SALIENCY_METHOD \
--mask "none" --dl --chkpt $CHECKPOINT
