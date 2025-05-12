# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions to load and to transform nii files.
"""
import logging
import numpy as np
import pandas as pd
import nibabel
from makedataset.utils import get_keys

def load_images(ni_participants_df, check=dict(), resampling=None, dtype=None):
    """
    Load images assuming paths contain a BIDS pattern to retrieve participant_id such /sub-<participant_id>/
    If id_regex is given, use it to retrieve participant id.
    Parameters
    ----------
    ni_participants_df : pandas DataFrame containing 'ni_path', path to the images to load
    check : dict, optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))
    resampling: float, factor to apply for resampling the images with nilearn.image.resample_img
    Returns
    -------
        ni_arr: ndarray, of shape (n_subjects, 1, image_shape). Shape should respect (n_subjects, n_channels, image_axis0, image_axis1, ...)
    Example
    -------
    >>> ni_filenames = ['/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR017/ses-V1/mri/mwp1sub-ICAAR017_ses-V1_acq-s03_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-ICAAR033/ses-V1/mri/mwp1sub-ICAAR033_ses-V1_acq-s07_T1w.nii',
    '/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-STARTRA160489/ses-V1/mri/mwp1sub-STARTRA160489_ses-v1_T1w.nii']
    >>> ni_participants_df = make_participants_df(ni_filenames)
    >>> ni_arr = load_images(ni_participants_df, check=dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5)))
    >>> ni_arr.shape
    (3, 1, 121, 145, 121)
    >>> ni_participants_df
      participant_id                                            ni_path
    0       ICAAR017  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489  /neurospin/psy/start-icaar-eugei/derivatives/c...
    """

    ni_imgs = [nibabel.load(ni_filename) for ni_filename in ni_participants_df.ni_path]
    ref_img = ni_imgs[0]
    # Check
    if 'shape' in check:
        assert ref_img.get_data().shape == check['shape']
    if 'zooms' in check:
        assert ref_img.header.get_zooms() == check['zooms']
    assert np.all([np.all(img.affine == ref_img.affine) for img in ni_imgs])
    assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in ni_imgs])

    ## Load image subjects x chanels (1) x image
    # Resampling
    if resampling is not None:
        from nilearn.image import resample_img
        target_affine =ref_img.affine[:3 ,:3] * resampling
        ni_arr = np.stack([np.expand_dims(resample_img(img, target_affine).get_data(), axis=0)
                           for img in ni_imgs])
    else:
        ni_arr = np.stack([np.expand_dims(img.get_data(), axis=0) for img in ni_imgs])

    if dtype is not None: # convert the np type
        ni_arr = ni_arr.astype(dtype)
    return ni_arr

def load_images_with_aims(ni_participants_df, check=dict(), dtype=None, stored_data=False):
    """
    Same function as load_images but with aims to load images instead of nibabel

    """
    from soma import aims

    logger = logging.getLogger("load_images_with_aims")

    ni_imgs = [aims.read(ni_filename) for ni_filename in ni_participants_df.ni_path]
    ref_img = ni_imgs[0]
    shape = np.array(ref_img.header()["volume_dimension"])
    voxel_size = np.array(ref_img.header()["voxel_size"])
    transformation = np.array(ref_img.header()["transformations"][-1])
    storage = np.array(ref_img.header()["storage_to_memory"])
    logger.debug(f"Tansformation of the ref image : {transformation}")
    logger.debug(f"Storage of the ref image : {storage}")
    if "shape" in check:
        assert np.all(shape[:3] == check['shape']), \
            print(f"Ref image does not have the right shape : {shape} / {check['shape']}")
    if "voxel_size" in check:
        assert np.all(voxel_size[:3] == check["voxel_size"]), \
            print(f"Ref image does not have the right voxel size : {voxel_size} / {check['voxel_size']}")
    if "transformation" in check:
        assert np.all(transformation == check["transformation"]), \
            print(f"Ref image does not have the right transformation : {transformation} / {check['transformation']}")
    if "storage" in check:
        assert np.all(storage == check["storage"]), \
            print(f"Ref image does not have the right storage : {storage} / {check['storage']}")

    assert np.all([np.all(np.array(img.header()["voxel_size"]) == voxel_size) for img in ni_imgs]), \
        print(f"All the images do not have the same voxel size {voxel_size}")
    assert np.all([np.all(np.array(img.header()["volume_dimension"]) == shape) for img in ni_imgs]), \
        print(f"All the images do not have the same shape {shape}")
    assert np.all([np.all(np.array(img.header()["transformations"][-1]) == transformation) for img in ni_imgs]), \
        print(f"All the images do not have the same transformation {transformation}")
    assert np.all([np.all(np.array(img.header()["storage_to_memory"]) == storage) for img in ni_imgs]), \
        print(f"All the images do not have the same storage {storage}")
    
    if stored_data:
        ni_imgs = list(map(lambda v: get_stored_data_with_aims(v), ni_imgs))

    ni_arr = np.stack([np.expand_dims(np.squeeze(np.asarray(img)), axis=0) for img in ni_imgs])

    if dtype is not None: # convert the np type
        ni_arr = ni_arr.astype(dtype)
    return ni_arr

def get_stored_data_with_aims(volume, background=0):

    from soma import aims, aimsalgo

    # FIXME : vérifier si la transformation fonctionne correctement (facteur 1.5 ?)
    # Transformation
    storage2memory = aims.AffineTransformation3d(volume.header()["storage_to_memory"])
    translation = np.array([volume.header()["storage_to_memory"][i] for i in range(3, 12, 4)])
    voxel_size = np.array(volume.header()["voxel_size"])[:3]
    storage2memory.setTranslation(translation *voxel_size)
    transformation = aims.AffineTransformation3d(storage2memory).inverse()

    # Set Resampler
    resampler = aims.ResamplerFactory_S16().getResampler(0) # Nearest-neghbours resampler
    resampler.setDefaultValue(background) # set background to 0
    resampler.setRef(volume) # volume to resample
    resampled_volume = resampler.doit(transformation, *volume.shape[:3], voxel_size)
    return resampled_volume


def img_to_array(img_filenames, check_same_referential=True, expected=dict()):
    """
    Convert nii images to array (n_subjects, 1, , image_axis0, image_axis1, ...)
    Assume BIDS organisation of file to retrive participant_id, session and run.

    Parameters
    ----------
    img_filenames : [str]
        path to images

    check_same_referential : bool
        if True (default) check that all image have the same referential.

    expected : dict
        optional dictionary of parameters to check, ex: dict(shape=(121, 145, 121), zooms=(1.5, 1.5, 1.5))

    Returns
    -------
        imgs_arr : array (n_subjects, 1, , image_axis0, image_axis1, ...)
            The array data structure (n_subjects, n_channels, image_axis0, image_axis1, ...)

        df : DataFrame
            With column: 'participant_id', 'session', 'run', 'path'

        ref_img : nii image
            The first image used to store referential and all information relative to the images.

    Example
    -------
    >>> from  nitk.image import img_to_array
    >>> import glob
    >>> img_filenames = glob.glob("/neurospin/psy/start-icaar-eugei/derivatives/cat12/vbm/sub-*/ses-*/mri/mwp1sub*.nii")
    >>> imgs_arr, df, ref_img = img_to_array(img_filenames)
    >>> print(imgs_arr.shape)
    (171, 1, 121, 145, 121)
    >>> print(df.shape)
    (171, 3)
    >>> print(df.head())
      participant_id session                                               path
    0       ICAAR017      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    1       ICAAR033      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    2  STARTRA160489      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    3  STARTLB160534      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...
    4       ICAAR048      V1  /neurospin/psy/start-icaar-eugei/derivatives/c...

    """

    df = pd.DataFrame([pd.Series(get_keys(filename)) for filename in img_filenames])
    imgs_nii = [nibabel.load(filename) for filename in df.ni_path]

    ref_img = imgs_nii[0]

    # Check expected dimension
    if 'shape' in expected:
        assert ref_img.get_fdata().shape == expected['shape']
    if 'zooms' in expected:
        assert ref_img.header.get_zooms() == expected['zooms']

    if check_same_referential: # Check all images have the same transformation
        assert np.all([np.all(img.affine == ref_img.affine) for img in imgs_nii])
        assert np.all([np.all(img.get_fdata().shape == ref_img.get_fdata().shape) for img in imgs_nii])

    assert np.all([(not np.isnan(img.get_fdata()).any()) for img in imgs_nii])
    # Load image subjects x channels (1) x image
    imgs_arr = np.stack([np.expand_dims(img.get_fdata(), axis=0) for img in imgs_nii])

    return imgs_arr, df, ref_img
