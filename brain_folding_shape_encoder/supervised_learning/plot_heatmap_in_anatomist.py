#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

from soma import aims, aimsalgo
import anatomist.direct.api as anatomist
from soma.qt_gui.qt_backend import Qt

def get_heatmap(db, test):
    path_to_xai = "/home/pa267054/scripts/xai"
    arr = np.load(os.path.join(path_to_xai, "transfer_learning", db, f"PatchOcclusion_classifier_epoch-99_{test}_test.npy"))
    heatmap = arr[:, 4:-4, :]
    # scaling
    mini, maxi = np.min(heatmap), np.max(heatmap)
    heatmap[heatmap > 0] = heatmap[heatmap > 0] / maxi
    heatmap[heatmap < 0] = heatmap[heatmap < 0] / np.abs(mini)
    
    print(f"heatmap max: {maxi} / min: {mini}")
    
    if db == "asd":
        vol_skel = aims.read(os.path.join(path_to_xai, "abide1", "Fresampled_skeleton_sub-51352_ses-1.nii.gz"))
    else:
        vol_skel = aims.read(os.path.join(path_to_xai, "cnp", "Fresampled_skeleton_sub-10159_ses-1.nii.gz"))
    storage2memory = aims.AffineTransformation3d(vol_skel.header()["storage_to_memory"])
    translation = np.array([vol_skel.header()["storage_to_memory"][i] for i in range(3, 12, 4)])
    voxel_size = np.array(vol_skel.header()["voxel_size"])[:3]
    storage2memory.setTranslation(translation *voxel_size)
    transformation = aims.AffineTransformation3d(storage2memory).inverse()

    # Set Resampler
    resampler = aims.ResamplerFactory_S16().getResampler(0) # Nearest-neghbours resampler
    resampler.setDefaultValue(0) # set background to 0
    resampler.setRef(vol_skel) # volume to resample
    resampled_skel = resampler.doit(transformation, *vol_skel.shape[:3], voxel_size)
    
    vol = aims.Volume(resampled_skel.getSize(), "FLOAT")
    vol.copyHeaderFrom(resampled_skel.header())
    arr = np.asarray(vol)
    nonzeros = np.nonzero(heatmap)
    for x, y, z in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
        arr[x, y, z, 0] = heatmap[x, y, z]
    return vol

def get_mesh(which="all"):
    lpath = aims.carto.Paths.findResourceFile(
        'models/models_2008/descriptive_models/segments'
        '/global_registered_spam_left/meshes/Lspam_model_meshes_1.arg')
    rpath = aims.carto.Paths.findResourceFile(
        'models/models_2008/descriptive_models/segments'
        '/global_registered_spam_right/meshes/Rspam_model_meshes_1.arg')

    l_graph = aims.read(lpath)
    r_graph = aims.read(rpath)

    all_meshes = None
    if which in ["left", "all"]:
        for vertex in l_graph.vertices():
            mesh = vertex.get("aims_Tmtktri")
            if mesh is not None:
                if all_meshes is None:
                    all_meshes = mesh
                else:
                    aims.SurfaceManip.meshMerge(all_meshes, mesh)
    if which in ["right", "all"]:
        for vertex in r_graph.vertices():
            mesh = vertex.get("aims_Tmtktri")
            if mesh is not None:
                if all_meshes is None:
                    all_meshes = mesh
                else:
                    aims.SurfaceManip.meshMerge(all_meshes, mesh)
    g_to_tal = aims.GraphManip.talairach(l_graph)
    all_meshes.header()['referentials'] \
        = aims.vector_STRING([aims.StandardReferentials.acPcReferentialID()])
    all_meshes.header()['transformations'] \
        = aims.vector_vector_FLOAT([g_to_tal.toVector()])
    all_meshes.header()['voxel_size'] = l_graph['voxel_size']
    return all_meshes

if __name__ == "__main__":
    for db in ("asd", "bd", "scz"):
        for test in ("internal", "external"):
            print(db , test)
            vol = get_heatmap(db, test)

    
    db = "asd"
    test = "external"
    
    # initialize Anatomist
    a = anatomist.Anatomist()
    
    # load mesh of the SPAM model
    lmesh= get_mesh(which="left")
    almesh = a.toAObject(lmesh)
    almesh.applyBuiltinReferential()
    rmesh= get_mesh(which="right")
    armesh = a.toAObject(rmesh)
    armesh.applyBuiltinReferential()
    a.execute('SetMaterial', objects=[almesh, armesh], diffuse=[1., 1., 1., 0.8])
    # load a volume in anatomist
    vol = get_heatmap(db, test)
    avol = a.toAObject(vol)
    avol.applyBuiltinReferential()
    avol.setPalette('cold_white_hot', zeroCentered1=True, minVal=0.,
                    maxVal=1, absoluteMode=True)

    # fusion between mesh and volume
    lfusion = a.fusionObjects(objects=[avol, almesh], method="Fusion3DMethod")
    a.execute("Fusion3DParams", object=lfusion,
              method="line", submethod="mean", depth=4, step=1)
    rfusion = a.fusionObjects(objects=[avol, armesh], method="Fusion3DMethod")
    a.execute("Fusion3DParams", object=rfusion,
              method="line", submethod="mean", depth=4, step=1)          
    # add to window in anatomist
    lwin = a.createWindow('3D')
    lwin.addObjects(lfusion)
    lwin.windowConfig(cursor_visibility=0)
    rwin = a.createWindow('3D')
    rwin.addObjects(rfusion)
    rwin.windowConfig(cursor_visibility=0)
    qapp = Qt.QApplication.instance()
    
    view_quaternion = {"left": [0.5, 0.5, 0.5, 0.5], "right": [0.5, -0.5, -0.5, 0.5]}
    for side, view in view_quaternion.items():
        lwin.camera(view_quaternion=view)
        rwin.camera(view_quaternion=view)
        lwin.snapshot(filename=f"/home/pa267054/scripts/xai/transfer_learning/{db}/xai_{db}_test-{test}_hemi-left_view-{side}.png", width=1920, height=1080)
        rwin.snapshot(filename=f"/home/pa267054/scripts/xai/transfer_learning/{db}/xai_{db}_test-{test}_hemi-right_view-{side}.png", width=1920, height=1080)
    qapp.exec()

    del win, fusion, avol, amesh 
