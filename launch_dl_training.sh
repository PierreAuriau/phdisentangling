################
## PARAMETERS ##
################

#Directories
ROOT="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/root"
CHK="20230302_gaus_conv_scz"
CHK_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/models/${CHK}"

#Experience
PREPROC="skeleton" # "vbm", "quasi_raw" or "skeleton"
PB="scz" # "scz",  "asd" or "bipolar"

#Model
NET="densenet121" #"densenet121",  "resnet18" or "alexnet"
NB_FOLDS=3
NB_EPOCHS=100
BATCH_SIZE=32
LR=1e-4
SCHEDULER=0.3
STEP_SIZE=10
SAMPLER="random"

FR_SAVE=10
NB_CPU_WORKERS=16

# Outputs filename
EXP_NAME="${NET}_${PREPROC}_${PB}"
#OUT_NAME="${NET}_${PREPROC}_${PB}"

##############
## COMMANDS ##
##############

# Update PythonPath
source /home/pa267054/.bashrc

# Activate environment
source /home_local/pa267054/skelediag/bin/activate

# Training
python3 /neurospin/dico/pauriau/git/SMLvsDL/dl_training/main.py --root $ROOT --checkpoint_dir $CHK_DIR --preproc $PREPROC \
--exp_name $EXP_NAME --pb $PB --nb_folds $NB_FOLDS --net $NET --batch_size $BATCH_SIZE \
--lr $LR --gamma_scheduler $SCHEDULER --sampler $SAMPLER --nb_epochs $NB_EPOCHS \
--step_size_scheduler $STEP_SIZE --nb_epochs_per_saving $FR_SAVE \
--num_cpu_workers $NB_CPU_WORKERS --cuda 1 --train --test --folds 2
