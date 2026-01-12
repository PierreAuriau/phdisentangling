#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd

from soma import aims
import anatomist.direct.api as anatomist
from soma.qt_gui.qt_backend import Qt

path_to_dataframe = "/home/pa267054/scripts/xai/local_models/%s_dlmodel_test_by_brain_area.csv"

path_to_json = "/home/pa267054/scripts/xai/local_models/sulci_regions_gridsearch.json"

mapping_area = {
    "FCLp-subsc-FCLa-INSULA": "F.C.L.p.-subsc.-F.C.L.a.-INSULA.",
    "FCMpost-SpC": "F.C.M.post.-S.p.C.",
    "FColl-SRh": "F.Coll.-S.Rh.",
    "FIP": "F.I.P.",
    "FPO-SCu-ScCal": "F.P.O.-S.Cu.-Sc.Cal.",
    "Lobule_parietal_sup": "Lobule_parietal_sup.",
    "ORBITAL": "S.Or.",
    "SC-SPeC": "S.C.-S.Pe.C.",
    "SC-SPoC": "S.C.-S.Po.C.",
    "SC-sylv": "S.C.-sylv.",
    "SFinf-BROCA-SPeCinf": "S.F.inf.-BROCA-S.Pe.C.inf.",
    "SFint-FCMant": "S.F.int.-F.C.M.ant.",
    "SFint-SR": "S.F.int.-S.R.",
    "SFinter-SFsup": "S.F.inter.-S.F.sup.",
    "SFmarginal-SFinfant":"S.F.marginal-S.F.inf.ant.",
    "SFmedian-SFpoltr-SFsup": "S.F.median-S.F.pol.tr.-S.F.sup.",
    "SOr-SOlf": "S.Or.-S.Olf.",
    "SPeC": "S.Pe.C.",
    "SPoC": "S.Po.C.",
    "STi-SOTlat": "S.T.i.-S.O.T.lat.",
    "STi-STs-STpol": "S.T.i.-S.T.s.-S.T.pol.",
    "STsbr": "S.T.s.br.",
    "STs": "S.T.s.",
    "ScCal-SLi": "Sc.Cal.-S.Li.",
    "SsP-SPaint": "S.s.P.-S.Pa.int.",
}

def get_sulci(region, regions):
    region = region.replace("CINGULATE", "CINGULATE.")
    region = region.replace("ORBITAL", "S.Or.")
    list_sulci = list(regions['brain'][f"{region}"].keys())
    list_sulci = [x.replace("paracingular.", "S.F.int.") for x in list_sulci]
    return list_sulci

def get_dataframe(path_to_dataframe, path_to_json, mapping_area, test="internal"):
    df = pd.read_csv(path_to_dataframe)
    regions = json.load(open(path_to_json, "r"))
    df2display = df.loc[df["set"] == f"{test}_test", ["area", "roc_auc"]].groupby("area").describe()
    df2display = df2display.droplevel(level=0, axis=1)
    df2display = df2display.rename(columns={"mean": "roc_auc"})
    df2display = df2display.drop(columns=["count", "25%", "50%", "75%"])
    df2display = df2display.reset_index()
    
    df2display["side"] = df2display["area"].str.split("_").str[-1]
    df2display["region"] = df2display["area"]
    for key in mapping_area.keys():
        df2display["region"] = df2display["region"].str.replace(key, mapping_area[key])
    df2display["sulcus"] = df2display["region"].apply(lambda x: get_sulci(x, regions))
    df2display = df2display.explode("sulcus")
    return df2display
    
def get_spam_graph(df, which="all"):
    lpath = aims.carto.Paths.findResourceFile(
        'models/models_2008/descriptive_models/segments'
        '/global_registered_spam_left/meshes/Lspam_model_meshes_1.arg')
    rpath = aims.carto.Paths.findResourceFile(
        'models/models_2008/descriptive_models/segments'
        '/global_registered_spam_right/meshes/Rspam_model_meshes_1.arg')
    
    if which == "left":
        graph = aims.read(lpath)
    else:
        graph = aims.read(rpath)
    
    for vertex in graph.vertices():
        vname = vertex.get("name")
        if vname in df["sulcus"].values:
            vertex["roc_auc"] = df.loc[df["sulcus"] == vname, "roc_auc"].max()
        else:
            vertex["roc_auc"] = 0
    return graph

if __name__ == "__main__":
    db = "scz"
    test = "external"
    
    df = get_dataframe(path_to_dataframe % db, path_to_json, mapping_area, test=test)
    
    # initialize Anatomist
    a = anatomist.Anatomist()
    
    # load graph of the SPAM model
    lgraph = get_spam_graph(df, which="left")
    algraph = a.toAObject(lgraph)
    algraph.setColorMode(algraph.PropertyMap)
    algraph.setColorProperty('roc_auc')
    algraph.notifyObservers()
    algraph.setMaterial(diffuse=[1., 1., 1., 0.8])
    algraph.setPalette("Yellow-red-fusion",
                        minVal=0.55, maxVal=0.62,
                        absoluteMode=True)
                        
    rgraph = get_spam_graph(df, which="right")
    argraph = a.toAObject(rgraph)
    argraph.setColorMode(argraph.PropertyMap)
    argraph.setColorProperty('roc_auc')
    argraph.notifyObservers()
    argraph.setMaterial(diffuse=[1., 1., 1., 0.8])
    argraph.setPalette("Yellow-red-fusion",
                        minVal=0.55, maxVal=0.62,
                        absoluteMode=True)
              
    # add to window in anatomist
    view_quaternion = {"left": [0.5, 0.5, 0.5, 0.5], "right": [0.5, -0.5, -0.5, 0.5]}
    
    lwin = a.createWindow('3D')
    lwin.addObjects(algraph)
    lwin.windowConfig(cursor_visibility=0)
    
    rwin = a.createWindow('3D')
    rwin.addObjects(argraph)
    rwin.windowConfig(cursor_visibility=0)
    
    for side, view in view_quaternion.items():
        lwin.camera(view_quaternion=view)
        rwin.camera(view_quaternion=view)
        lwin.snapshot(filename=f"/home/pa267054/scripts/xai/local_models/{db}_test-{test}_hemi-left_view-{side}.png", width=1920, height=1080)
        rwin.snapshot(filename=f"/home/pa267054/scripts/xai/local_models/{db}_test-{test}_hemi-right_view-{side}.png", width=1920, height=1080)
    qapp = Qt.QApplication.instance()
    qapp.exec()

    del win, fusion, agraph
