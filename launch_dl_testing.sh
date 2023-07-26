# Parameters
ROOT="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/root"
MODEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/models"

CHK_DIR="${MODEL_DIR}/20230111_schizophrenic_wo_ventricle_skeletons/sleek-sweep-2"
PREPROC="skeleton" # "vbm", "quasi_raw" or "skeleton"
NET="densenet121" #"densenet121",  "resnet18" or "alexnet"
PB="scz" # "scz",  "asd" or "bipolar"

EXP_NAME="${PB}_${PREPROC}_${NET}"
OUT_NAME="${PB}_${PREPROC}_${NET}_SequentialSampler"

NB_EPOCHS=100
BATCH_SIZE=8
SAMPLER="sequential"
NB_FOLDS=3
NUM_WORKERS=8

# Command
python3 /neurospin/dico/pauriau/git/SMLvsDL/dl_training/main.py --root $ROOT  \
--preproc $PREPROC --checkpoint_dir $CHK_DIR \
--exp_name $EXP_NAME --outfile_name $OUT_NAME \
--pb $PB --nb_folds $NB_FOLDS --net $NET \
--batch_size $BATCH_SIZE --lr 1e-4 --gamma_scheduler 0.4 \
--nb_epochs $NB_EPOCHS --sampler $SAMPLER  \
--num_cpu_workers $NUM_WORKERS \
--cuda 0 --test