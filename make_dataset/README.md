# Make Datasets
This folder contains all the scripts to generate skeleton datasets.
In the `makedataset` directory, there is a little package `makedataset` with all the functions to create a numpy array and a participant dataframe from skeleton volumes.
In the `cohorts` directory, there are scripts to create datasets for several clinical studies. The scripts create the skeleton volumes from the skeleton graphs (the outputs of the *[Brainvisa/Morphologist pipeline](https://brainvisa.info/web/morphologist.html)*) thanks to the *[deep_folding toolbox](https://github.com/neurospin/deep_folding)* and then create a numpy array with all the skeleton volumes and its associated participant dataframe. These numpy arrays can be used as inputs of Deep Learning models.

# Prerequesites
To process skeleton graphs, you need to install the Brainvisa container (see : <https://brainvisa.info/web/download.html>) and the deep_folding repertory (see : <https://github.com/neurospin/deep_folding>). 
``` shell
# Launch brainvisa container
brainvisa/bin/bv bash
# Create virtual environnement
virtualenv --python=python3 --system-site-packages myenv
source myenv/bin/activate
# Install deep_folding
git clone https://github.com/neurospin/deep_folding.git
cd deep_folding
pip3 install -e .
# Install makedataset
git clone https://github.com/PierreAuriau/phdisentangling.git
cd phdisentangling/make_dataset
pip3 install -e .
```
# Process 

To launch the pre-processing of the clinical studies, you need to go into the study folder and launch the `skeleton_make_dataset.py` script :
``` shell
cd cohorts/<name_of_the_study>
python skeleton_make_dataset.py
```
## deep_folding
The first steps is to generate skeleton images from graph files with the deep_folding toolbox :
1. Generate transformation files to put skeleton images in the MNI-ICBM2009c template (affine transformation).
2. Generate raw skeleton images from the graph files.
3. Remove ventricle from skeleton images thanks to the Morphologist automatic labelling.
4. Resample the skeleton images and apply computed transformations.

These first steps can also be launched with the bash script `launch_deep_folding_scripts.sh`:
``` shell
./launch_deep_folding_scripts.sh
```

## skeleton to array
The second step is to create the numpy array with the generated images. In parallel, a csv file is create with all the metadata about the participants of the study.
Subjects could be selected according to a quality check file.

## Other preprocessing
For the VBM, quasi-raw or Freesurfer preprocessings, there are scripts named `<preprocessing_name>_make_dataset.py` that gather all the files into an array along with a participant dataframe.


# Useful links
* Brainvisa : <https://brainvisa.info/web/>
* deep_folding : <https://github.com/neurospin/deep_folding>
* ns-datasets : <https://github.com/neurospin/ns-datasets> (1st version of the make_dataset_utils.py script)
