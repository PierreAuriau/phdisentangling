# Make Datasets
This folder contains all the scripts to generate skeleton datasets.
In the `src` directory, there is a litlle package `makedataset` with all the functions to create a numpy array and a participant dataframe from skeleton volumes.
In the `cohorts` directory, the example scripts for some clinical studies. The scripts create a numpy array with all the skeleton images from the skeleton graphs which are the outputs of the Brainvisa/Morphologist pipeline. These numpy arrays serve as inputs of Deep Learning models.

# Prerequesites
To process skeleton graphs, you need to install the Brainvisa container (see : <https://brainvisa.info/web/download.html>) and the deep_folding repertory (see : <https://github.com/neurospin/deep_folding>). 
``` shell
# Launch brainvisa container
brainvisa/bin/bv bash
# Create virtual environnement
python3 -m venv --system-site-packages myenv
source myenv/bin/activate
# Install deep_folding
git clone https://github.com/neurospin/deep_folding.git
cd deep_folding
pip install -e .
# Install makedataset
git clone https://github.com/PierreAuriau/phdisentangling.git
cd phdisentangling/make_dataset
pip install -e .
```
# Process 

## deep_folding
The first steps is to generate skeleton images from graph files with the deep_folding toolbox :
1. Generate transformation files to put skeleton images in the MNI-ICBM2009c template (affine transformation).
2. Generate raw skeleton images from the graph files.
3. Remove ventricle from skeleton images thanks to the Morphologist automatic labelling.
4. Resample the skeleton images and apply computed transformations.

``` shell
./launch_deep_folding_scripts.sh
```

## skeleton to array
The second step is to create the numpy array with the generated images. In parallel, a csv file is create with all the metadata about the participants of the study.
Subjects could be selected according to a quality check file. 
``` shell
python skeleton_make_dataset.py
```

The two steps are gathered in a unique script : `<study>_skeleton_make_dataset.py` (in process)

## other scripts
* `make_skeleton_summary.py` : check which subjects do not have outputs of the Morphologist and deep_folding pipelines.
* `check_voxel_size.py` : check the voxel size of all the raw images. Images with a resolution under 1mm do not pass through Morphologist pipeline proprely. The goal is to know which images should be downsampled before passing through the pipeline. That is why we directly select images which have passed the CAT12VBM QC in order not to pass a second time the image in the pipeline if the image is not good. Only images in the BIOBD studies have problematic resolutions.

# Useful links
* Brainvisa : <https://brainvisa.info/web/>
* deep_folding : <https://github.com/neurospin/deep_folding>
* ns-datasets : <https://github.com/neurospin/ns-datasets> (1st version of the make_dataset_utils.py script)

