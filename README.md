I gathered on this repository all the code I developped during my PhD, entitled:

# Identification of neurodevelopmental variability in brain anatomical images

Abstract:


## Repository organization

* ```make_dataset```: scripts for the creation of datasets for classification. For each dataset, a numpy array is created with all the skeletons of the subjects and a tsv file is created with all the meta-data. I developped the scripts in particular for the morphologist preprocessing (skeletons).
* ```brain_folding_shape_encoder```: developping an encoder able to extract relevant representations of brain folding (chapter 2).
* ```brain_image_disentangling```: developping models to isolate the neurodevelopmental contribution to brain anatomy (chapter 3).
