#/bin/bash

## PARAMETERS ##
STUDY="bsnip1"

SRC_DIR="/neurospin/psy_sbox/${STUDY}/derivatives/morphologist-2021/subjects"
SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/raw/wo_ventricles"

NB_SUBJECT="all" #"all"

# Transforms
TRANSFORM_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/transforms"

SIDE="F"
PATH_TO_GRAPH="t1mri/ses-*_acq-*_run-*/default_analysis/folds/3.1"

VOXEL_SIZE=1.5
RESAMPLED_SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/${VOXEL_SIZE}mm/wo_ventricles"

SRC_FILENAME="skeleton_generated_"
OUTPUT_FILENAME="resampled_skeleton_"


## SCRIPTS ##
DEEP_FOLDING_DIR="/neurospin/dico/pauriau/git/deep_folding/deep_folding"

# Transforms
#python3 brainvisa/generate_ICBM2009c_transforms.py --src_dir $SRC_DIR --output_dir $TRANSFORM_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT --verbose --session --run

# Generate skeletons
#python3 brainvisa/generate_skeletons.py --src_dir $SRC_DIR --output_dir $SKELETON_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT -vvv -b

# Resample skeletons
python3 ${DEEP_FOLDING_DIR}/brainvisa/resample_files.py --src_dir $SKELETON_DIR --output_dir $RESAMPLED_SKELETON_DIR --side $SIDE --input_type "skeleton" --transform_dir $TRANSFORM_DIR --out_voxel_size $VOXEL_SIZE -e $OUTPUT_FILENAME -f $SRC_FILENAME -n ${NB_SUBJECT} -a
