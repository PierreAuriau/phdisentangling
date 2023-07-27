#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Pipeline to create skeleton dataset for the BIOBD study
To launch the script, you need to be in the brainvisa container and to install deep_folding
See the repository: https://github.com/neurospin/deep_folding

"""

import os
import subprocess
import json
import pandas as pd
import numpy as np
# deep_folding
from deep_folding.brainvisa import generate_skeletons 
from deep_folding.brainvisa import generate_ICBM2009c_transforms 
from deep_folding.brainvisa import remove_ventricle
from deep_folding.brainvisa import resample_files 
# Make dataset
from make_skeleton_summary import make_morphologist_summary, \
                                  make_deep_folding_summary, \
                                  merge_skeleton_summaries
from make_dataset_utils import skeleton_nii2npy


######## TO DELETE ########
if os.path.exists('/home/pa267054/neurospin'):
    neurospin = '/home/pa267054/neurospin'
else:
    neurospin = '/neurospin'
###########################


study = 'biobd'

# Study directories
study_dir = os.path.join(neurospin, 'psy_sbox', study)
morpho_dir = os.path.join(study_dir, 'derivatives', 'morphologist-2021')

# Output directory where to put all generated files
output_dir = os.path.join(neurospin, "psy_sbox", "analyses", "202205_predict_neurodev", "data", "skeleton", study)

# Path towards the deep_folding scripts
deep_folding_dir = os.path.join(neurospin, "dico", "pauriau", "git", "deep_folding", "brainvisa")

# Parameters
voxel_size = "1.5" #resampled voxel size
junction = "thin" # "wide" or "thin"
side = "F" # "F", "L" or "R"
bids = True

labelling_session = "deepcnn_session_auto"
skeleton_filename = "skeleton_generated"
without_ventricle_skeleton_filename = "skeleton_without_ventricle"
resampled_skeleton_filename = "resampled_skeleton"

# For debugging, put parallel=False, number_subjects=1 and verbosity = True
parallel = True
number_subjects = "all" # all subjects
verbosity = False

# Morphologist directory containing the subjects as subdirectories
src_dir = os.path.join(morpho_dir, 'subjects')

# Output directory where to put raw skeleton
skeleton_dir = os.path.join(output_dir, "raw")

# Relative path to graph for biobd dataset
path_to_graph = "t1mri/*/default_analysis/folds/3.1"

# Output directory where to put transform files
transform_dir = os.path.join(output_dir, "transforms")

# Output directory where to put skeleton files without ventricles
without_ventricle_skeleton_dir = os.path.join(output_dir, "without_ventricle")

# Output directory where to put resample skeleton files
resampled_skeleton_dir = os.path.join(output_dir, f"{voxel_size}mm")


# Morphologist summary
morphologist_df = make_morphologist_summary(morpho_dir=morpho_dir, 
                                            dataset_name=study,
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            labelling_session=labelling_session,
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

# DEEP FOLDING

# Generate transform files
argv = ["--src_dir", src_dir, 
        "--output_dir", transform_dir, 
        "--path_to_graph", path_to_graph,
        "--side", side,
        "--nb_subjects", number_subjects,
        "--qc_file", os.path.join(output_dir, "metadata", f"{study}_morphologist_summary.tsv")]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")
generate_ICBM2009c_transforms.main(argv)
"""
process = subprocess.run([sys.executable, os.path.join(deep_folding_dir, "generate_ICBM2009c_transforms.py")] + argv, 
                         check=True, capture_output=True, text=True)
print("OUTPUT\n", process.stdout)
print("\nERRORS\n", process.stderr)
"""
# Generate skeletons from graphs
argv = ["--src_dir", src_dir, 
        "--output_dir", skeleton_dir, 
        "--path_to_graph", path_to_graph,
        "--side", side,
        "--junction", junction,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")
    
generate_skeletons.main(argv)
"""
process = subprocess.run([sys.executable, os.path.join(deep_folding_dir, "generate_skeletons.py")] + argv, 
                         check=True, capture_output=True, text=True)
print("OUTPUT\n", process.stdout)
print("\nERRORS\n", process.stderr)
"""
# Remove ventricles from the skeletons
argv = ["--skeleton_dir", skeleton_dir, 
        "--output_dir", without_ventricle_skeleton_dir, 
        "--path_to_graph", path_to_graph,
        "--morpho_dir", src_dir,
        "--side", side,
        "--labelling_session", labelling_session,
        "--skeleton_filename", skeleton_filename,
        "--output_filename", without_ventricle_skeleton_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")
    
remove_ventricle.main(argv)
"""
process = subprocess.run([sys.executable, os.path.join(deep_folding_dir, "remove_ventricle.py")] + argv, 
                         check=True, capture_output=True, text=True)
print("OUTPUT\n", process.stdout)
print("\nERRORS\n", process.stderr)
"""
# Resample skeletons in the ICBM2099c template
argv = ["--src_dir", without_ventricle_skeleton_dir, 
        "--output_dir", resampled_skeleton_dir, 
        "--side", side,
        "--input_type", "skeleton",
        "--transform_dir", transform_dir,
        "--out_voxel_size", voxel_size,
        "--src_filename", without_ventricle_skeleton_filename,
        "--output_filename", resampled_skeleton_filename,
        "--nb_subjects", number_subjects]
if parallel:
    argv.append("--parallel")
if bids:
    argv.append("--bids")
if verbosity:
    argv.append("--verbosity")
resample_files.main(argv)
"""
process = subprocess.run([sys.executable, os.path.join(deep_folding_dir, "resample_files.py")] + argv, 
                         check=True, capture_output=True, text=True)
print("OUTPUT\n", process.stdout)
print("\nERRORS\n", process.stderr)
"""
# deep_folding summary
df_directories = {"skeleton": skeleton_dir,
                  "transform": transform_dir,
                  "skeleton_without_ventricle": without_ventricle_skeleton_dir,
                  "resampled_skeleton": resampled_skeleton_dir
                  }
deep_folding_df = make_deep_folding_summary(deep_folding_directories=df_directories, 
                                            side=side, 
                                            dataset_name=study, 
                                            path_to_save=os.path.join(output_dir, "metadata"),
                                            id_regex="sub-([^_/\.]+)",
                                            ses_regex="ses-([^_/\.]+)",
                                            acq_regex="acq-([^_/\.]+)",
                                            run_regex="run-([^_/\.]+)")

df = merge_skeleton_summaries(morphologist_df, deep_folding_df, 
                              dataset_name=study, 
                              path_to_save=os.path.join(output_dir, "metadata"))

# MAKE ARRAYS

# Parameters
stored_data=False
skeleton_size=False
regex = f"{side}resampled_skeleton_*.nii.gz"
nii_path = os.path.join(resampled_skeleton_dir, side, regex)
output_path = os.path.join(output_dir, "arrays", "with_tiv")

check = {"shape": (128, 152, 128), 
        "voxel_size": (1.5, 1.5, 1.5),
        "transformation": np.array([-1, 0, 0, 96, 0, -1, 0, 96, 0, 0, -1, 114, 0, 0, 0, 1]),
        "storage": np.array([-1, 0, 0, 127, 0, -1, 0, 151, 0, 0, -1, 127, 0, 0, 0, 1])
        }

# Phenotype
phenotype_filename = os.path.join(study_dir, 'participants.tsv')
phenotype = pd.read_csv(phenotype_filename, sep='\t')
phenotype = phenotype.rename(columns={"TIV": "tiv"})
assert phenotype["study"].notnull().values.all(), "study column in phenotype has nan values"
assert phenotype["site"].notnull().values.all(), "site column in phenotype has nan values"
assert phenotype["tiv"].notnull().values.all(), "tiv column in phenotype has nan values"

# Quality checks
vbm_qc = pd.read_csv(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc", "qc.tsv"), sep="\t")
map_subject_run = json.load(open(os.path.join(output_dir, "metadata", "mapping_all_subjects_run.json"), "r"))
for sbj in map_subject_run.keys():
    run = map_subject_run[sbj]["run"]   
    vbm_qc.loc[vbm_qc["participant_id"] == int(sbj), "run"] = run
vbm_qc["run"] = vbm_qc["run"].astype(int)
skel_qc = pd.read_csv(os.path.join(output_dir, "metadata", "qc_morphologist.tsv"), sep="\t")

qc = pd.merge(vbm_qc, skel_qc, how="outer", on=["participant_id", "session", "run"],
              validate="1:1", suffixes=("_vbm", "_skeleton"))
qc["qc_vbm"] = qc["qc_vbm"].fillna(0)
qc["qc_skeleton"] = qc["qc_skeleton"].fillna(0)
qc["qc"] = qc[["qc_vbm", "qc_skeleton"]].prod(axis=1).astype(int)
#qc["session"] = qc["session"].fillna(1)
#qc["session"] = qc["session"].astype(int)

# Array creation
skeleton_nii2npy(nii_path=nii_path, 
                  phenotype=phenotype, 
                  dataset_name=study, 
                  output_path=output_path, 
                  qc=qc, 
                  sep='\t', 
                  id_type=str,
                  check=check, 
                  side=side,
                  skeleton_size=skeleton_size,
                  stored_data=stored_data)