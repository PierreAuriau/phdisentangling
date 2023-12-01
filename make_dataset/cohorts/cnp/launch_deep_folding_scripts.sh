#/bin/bash

## PARAMETERS ##
STUDY="cnp"

SRC_DIR="/neurospin/psy_sbox/${STUDY}/derivatives/morphologist-2021/subjects"
#SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/raw/wo_ventricles"
SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/test/raw"

FOLDLABEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/foldlabels/${STUDY}/raw"
#SULCUSLABEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/sulcuslabels/${STUDY}/raw/without_ventricles"
SULCUSLABEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/sulcuslabels/${STUDY}/without_ventricles"

NB_SUBJECT="10" #"all"

# Transforms
TRANSFORM_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/transforms"

SIDE="F"
PATH_TO_GRAPH="t1mri/*/default_analysis/folds/3.1"
LABELLING_SESSION="deepcnn_session_auto"
PATH_TO_LABELLED_GRAPH="${PATH_TO_GRAPH}/${LABELLING_SESSION}"

VOXEL_SIZE=1.5
RESAMPLED_SKELETON_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/${STUDY}/${VOXEL_SIZE}mm/wo_ventricles"
RESAMPLED_SULCUSLABEL_DIR="/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/sulcuslabels/${STUDY}/${VOXEL_SIZE}mm/wo_ventricles"

SRC_FILENAME="skeleton_generated"
OUTPUT_FILENAME="resampled_skeleton"

DICO_SULCI="/neurospin/dico/pauriau/git/deep_folding/deep_folding/brainvisa/dico_sulci.json"

## SCRIPTS ##
DEEP_FOLDING_DIR="/neurospin/dico/pauriau/git/deep_folding/deep_folding"

# Transforms
#python3 brainvisa/generate_ICBM2009c_transforms.py --src_dir $SRC_DIR --output_dir $TRANSFORM_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT --verbose --session --run

# Generate skeletons
#python3 brainvisa/generate_skeletons.py --src_dir $SRC_DIR --output_dir $SKELETON_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT -vvv -b

# Resample skeletons
#python3 ${DEEP_FOLDING_DIR}/brainvisa/resample_files.py --src_dir $SKELETON_DIR --output_dir $RESAMPLED_SKELETON_DIR --side $SIDE --input_type "skeleton" --transform_dir $TRANSFORM_DIR --out_voxel_size $VOXEL_SIZE -e $OUTPUT_FILENAME -f $SRC_FILENAME -n ${NB_SUBJECT} -a

#Generate foldlabels
#python3 ${DEEP_FOLDING_DIR}/brainvisa/generate_foldlabels.py --src_dir $SRC_DIR --output_dir $FOLDLABEL_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --nb_subjects $NB_SUBJECT --bids

# Generate sulcuslabels
python3 ${DEEP_FOLDING_DIR}/brainvisa/generate_sulcuslabels.py --src_dir $SRC_DIR --output_dir $SULCUSLABEL_DIR --side $SIDE --path_to_graph $PATH_TO_GRAPH --labelling_session $LABELLING_SESSION  --dico_sulci $DICO_SULCI --nb_subjects $NB_SUBJECT --remove_ventricle --bids -v

# Resampling
#python3 ${DEEP_FOLDING_DIR}/brainvisa/resample_files.py --src_dir $SULCUSLABEL_DIR --output_dir $RESAMPLED_SULCUSLABEL_DIR --side $SIDE --input_type "foldlabel" --transform_dir $TRANSFORM_DIR --out_voxel_size $VOXEL_SIZE -e "resampled_sulcuslabel" -f "sulcuslabel_generated" -n $NB_SUBJECT -a
