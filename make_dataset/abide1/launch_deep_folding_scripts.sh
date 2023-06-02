#!/bin/bash

## PARAMETERS ##

STUDY="abide1"

SRC_DIR="/neurospin/psy_sbox/${STUDY}/derivatives/morphologist-2021/subjects"
OUTPUT_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}"

# Transforms
PATH_TO_GRAPH="t1mri/*/default_analysis/folds/3.1"
TRANSFORM_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/transforms"
# Skeletons
SKELETON_DIR="${OUTPUT_DIR}/raw"
# Skeletons without ventricles
LABELLING_SESSION="default_session_auto"
WOV_SKELETON_DIR="${OUTPUT_DIR}/without_ventricle"
# Resampled Skeletons
VOXEL_SIZE=1.5
RESAMPLED_SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/${VOXEL_SIZE}mm"
SKELETON_FILENAME="skeleton_generated"
WOV_SKELETON_FILENAME="skeleton_without_ventricle"
RESAMPLED_SKELETON_FILENAME="resampled_skeleton"

# Params
NB_SUBJECT="all"
SIDE="F"
# add -a for parallel execution

## SCRIPTS ##
DEEP_FOLDING_DIR="/neurospin/dico/pauriau/git/deep_folding/deep_folding"

# Transforms
#python3 ${DEEP_FOLDING_DIR}/brainvisa/generate_ICBM2009c_transforms.py --src_dir $SRC_DIR --output_dir $TRANSFORM_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT --bids -a

# Generate skeletons
#python3 ${DEEP_FOLDING_DIR}/brainvisa/generate_skeletons.py --src_dir $SRC_DIR --output_dir $SKELETON_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT --bids -v

# Remove ventricles
#python3 ${DEEP_FOLDING_DIR}/brainvisa/remove_ventricle.py --skeleton_dir $SKELETON_DIR --output_dir $WOV_SKELETON_DIR --morpho_dir $SRC_DIR --path_to_graph $PATH_TO_GRAPH --labelling_session $LABELLING_SESSION --skeleton_filename $SKELETON_FILENAME --output_filename $WOV_SKELETON_FILENAME --side $SIDE --nb_subjects $NB_SUBJECT --bids -a

# Resample skeletons
python3 ${DEEP_FOLDING_DIR}/brainvisa/resample_files.py --src_dir $WOV_SKELETON_DIR --output_dir $RESAMPLED_SKELETON_DIR --side $SIDE --input_type "skeleton" --transform_dir $TRANSFORM_DIR --out_voxel_size $VOXEL_SIZE --src_filename $WOV_SKELETON_FILENAME --output_filename $RESAMPLED_SKELETON_FILENAME  -n ${NB_SUBJECT} -a
