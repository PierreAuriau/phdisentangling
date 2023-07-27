import os
from soma import aims
import glob
import numpy as np
import pandas as pd
import re
import tqdm
import json

study = "bsnip1"
neurospin = "/neurospin"

id_type = str


study_dir = os.path.join(neurospin, "psy_sbox", study)
morpho_dir = os.path.join(study_dir, "derivatives", "morphologist-2021")

path_to_qc = os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_qc_2018", "qc.tsv")
nii_reg = os.path.join(morpho_dir, "subjects", "sub-*", "t1mri", "ses-*_acq-*_run-*", "*.nii.gz")
nii_files = glob.glob(nii_reg)

#QC
qc = pd.read_csv(path_to_qc, sep="\t")

qc["acq"] = None
with open(os.path.join(study_dir, "derivatives", "cat12-12.6_vbm_2018", "list_t1w.txt"), "r") as f:
          selected_files = f.readlines()
for file in selected_files:
    sbj = re.search("sub-([^_/]+)", file)
    ses = re.search("ses-([^_/]+)", file)
    acq = re.search("acq-([^_/]+)", file)
    assert np.all(qc.loc[qc["participant_id"] == sbj[1], ["session"]] == ses[1])
    if len(acq[1]) > 3:
        qc.loc[qc["participant_id"] == sbj[1], ["acq"]] = acq[1].replace(".", "")
    else:
        qc.loc[qc["participant_id"] == sbj[1], ["acq"]] = acq[1].replace(".", "0")

qc["session"] = qc["session"].replace("V1", "1")

qc.participant_id = qc.participant_id.astype(id_type)
unique_key_qc = []
if qc is not None and 'session' in qc:
	qc.session = qc.session.astype(id_type)
	unique_key_qc.append('session')
if qc is not None and 'run' in qc:
	qc.run = qc.run.astype(id_type)
	unique_key_qc.append('run')
if qc is not None and 'acq' in qc:
	qc.acq = qc.acq.astype(id_type)
	unique_key_qc.append('acq')
print(unique_key_qc)
# data to register
data = {"participant_id": [],
"session" : [],
"run" : [],
"acq" : [],
"nii_filename": [],
"voxel_size_0" : [],
"voxel_size_1" : [],
"voxel_size_2" : [],
"voxel_size_3" : [],
"under_1mm" : [],
"qc" : []
}

nb_files = len(nii_files)
for i, f in enumerate(nii_files):

	print(f"FILE {i}/{nb_files}")
	# Extract data
	
	# subject data
	data["nii_filename"].append(f) 
	
	participant_id = re.search("sub-([^/]+)", f)
	ses =  re.search("ses-([^_/]+)", f)
	run = re.search("run-([^_/]+)", f)
	acq = re.search("acq-([^_/]+)", f)
	
	data["participant_id"].append(participant_id[1])
	data["session"].append(ses[1])
	
	if run:
		data["run"].append(run[1])
	else:
		data["run"].append(None)
	if acq:
		data["acq"].append(acq[1])
	else:
		data["acq"].append(None)
	
	# voxel size
	img = aims.read(f)
	voxel_size = np.asarray(img.header()["voxel_size"])
	
	for i in range(len(voxel_size)):
		data[f"voxel_size_{i}"].append(voxel_size[i])
	
	if len(voxel_size) == 3:
		data[f"voxel_size_3"].append(None)
	
	# to be processed images
	if np.any(voxel_size < 1):
		data["under_1mm"].append(1)
	else:
		data["under_1mm"].append(0)
	
	#qc
	cond = (qc["participant_id"] == data["participant_id"][-1])
	for k in unique_key_qc:
		cond &= qc[k] == data[k][-1]
	
	qc_value = qc.loc[cond, ["qc"]].values
	print(f"Subject {participant_id[1]} / Session {ses[1]} / Acq {acq[1]} : {qc_value}")
	if qc_value.size == 1:
		data["qc"].append(int(qc_value.squeeze()))
	elif qc_value.size == 0:
		data["qc"].append(0)
	else:
		print(f"Error : the file {f} is not identified uniquely in the QC")
		break
	
		
df = pd.DataFrame(data)
df["to_be_downsampled"] = df["under_1mm"]*df["qc"]
df.to_csv(f"{study}_to_be_downsampled.csv")
print("Nb of sbj to be downsampled :", np.count_nonzero(df["to_be_downsampled"].values))
print("CSV saved")














