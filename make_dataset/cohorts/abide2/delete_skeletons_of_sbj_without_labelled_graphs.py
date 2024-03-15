import os
import pandas as pd

study = "abide2"
side = "F"
skeleton_dir = f"/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/{study}/raw/{side}"
transform_dir = f"/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/{study}/transforms/{side}"
summary = pd.read_csv(f"/neurospin/psy_sbox/analyses/202205_predict_neurodev/data/skeletons/{study}/metadata/{study}_morphologist_summary.tsv", sep="\t")

sbj2remove = summary[summary["qc"] == 0]

cnt = 0
cntbis = 0
for index, sbj in sbj2remove.iterrows():
    skeleton = f"{side}skeleton_generated_sub-{sbj['participant_id']}_ses-{sbj['session']}_run-{sbj['run']}.nii.gz"
    if os.path.exists(os.path.join(skeleton_dir, skeleton)):
        cnt += 1
        os.remove(os.path.join(skeleton_dir, skeleton))
        os.remove(os.path.join(skeleton_dir, skeleton + ".minf"))
    transform = f"{side}transform_to_ICBM2009c_sub-{sbj['participant_id']}_ses-{sbj['session']}_run-{sbj['run']}.trm"
    if os.path.exists(os.path.join(transform_dir, transform)):
        cntbis += 1
        os.remove(os.path.join(transform_dir, transform))
        os.remove(os.path.join(transform_dir, transform + ".minf"))
print(cnt)
print(cntbis)
    
