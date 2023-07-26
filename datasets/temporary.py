from datasets.open_bhb import OpenBHB
from datasets.clinical_multisites import SCZDataset
import os
import numpy as np
import logging


class TemporaryDataset(SCZDataset):
    @property
    def _train_val_test_scheme(self):
        self.logger.debug("Pickle for splits : 2")
        return "train_val_test_test-intra_scz2_stratified.pkl"

    def __str__(self):
        return "TemporaryDataset"

"""
    def _set_dataset_attributes(self):
        self._studies = ["schizconnect-vip", "bsnip", "cnp", "candi"]
        self._train_val_test_scheme = "train_val_test-intra_temporary_stratified.pkl"
        self._cv_scheme = None
        self._mapping_sites = "mapping_site_name-class_scz.pkl"
        self._npy_files = {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                           "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy",
                           "skeleton": "%s_t1mri_skeleton_data64.npy"}
        self._pd_files = {"vbm": "%s_t1mri_mwp1_participants.csv",
                          "quasi_raw": "%s_t1mri_quasi_raw_participants.csv",
                          "skeleton": "%s_t1mri_skeleton_participants.csv"}
        self._preproc_folders = {"vbm": "cat12vbm", "quasi_raw": "quasi_raw", "skeleton": "morphologist"}

    def _check_integrity(self):
        if self.scheme_name == "cv":
            raise NotImplementedError("No CV scheme implemented for BHB (yet).")
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))
        is_complete &= os.path.isfile(os.path.join(self.root, self._mapping_sites))
        dir_files = {
            "morphologist": ["%s_t1mri_skeleton_participants.csv", "%s_t1mri_skeleton_data64.npy"]
        }
        for (dir, files) in dir_files.items():
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, dir, file % db))
        return is_complete

    def _extract_metadata(self, df):
        self.logger = logging.getLogger("temporary")
        metadata = ["age", "sex", "site", "tiv"] + \
                   [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        if len(metadata) != 288:
            self.logger.warning("Missing meta-data values (%i != %i)" % (len(metadata), 288))
        if set(metadata) > set(df.keys()):
            self.logger.warning("Missing meta-data columns: {}".format(set(metadata) - set(df.keys)))
        if df[metadata].isna().sum().sum() != 0:
            self.logger.warning("NaN values found in meta-data")
        return df[metadata]

    def _extract_mask(self, df, unique_keys):
        # TODO: correct this hack in the final version
        df = df.copy()
        df.loc[df['run'].isna(), 'run'] = 1
        df.loc[df['session'].isna(), 'session'] = 1
        if df['run'].dtype == np.float:
            df['run'] = df['run'].astype(int)
        clinical_studies = ['BIOBD', 'BSNIP', 'SCHIZCONNECT-VIP', 'PRAGUE']
        df.loc[df['session'].eq('V1') & df['study'].isin(clinical_studies), 'session'] = 1
        df.loc[df['session'].eq('v1') & df['study'].isin(clinical_studies), 'session'] = 1

        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(np.bool)
        return mask

    def __str__(self):
        if self.fold is not None:
            return "TemporaryDataset-%s-%s-%s-%s" % (self.preproc, self.scheme_name, self.split, self.fold)
        return "TemporaryDataset-%s-%s-%s" % (self.preproc, self.scheme_name, self.split)
"""