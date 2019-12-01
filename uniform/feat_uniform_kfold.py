import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

# score on train data for informed uniform distro, which places uniform
# distribution on all classes that are in the same galactic subset.
def score_train():

    abs_path = "/modules/cs342/Assignment2/"
    metadata = pd.read_csv(abs_path+'training_set_metadata.csv')

    preds = pd.DataFrame(metadata.object_id, columns=["object_id"])    
    gal_classes = sorted(metadata[metadata.hostgal_photoz==0].target.unique())
    ext_classes = sorted(metadata[metadata.hostgal_photoz!=0].target.unique())

    gal_prob = 1.0/(len(gal_classes))
    ext_prob = 1.0/(len(ext_classes))    

    # same proabability of belonging to each class
    for obj_class in sorted(metadata.target.unique()):
        if obj_class in gal_classes:
            pred_class = [gal_prob if h==0 else 0 for h in metadata.hostgal_photoz]    
        else:
            pred_class = [ext_prob if h!=0 else 0 for h in metadata.hostgal_photoz]
        preds['class_'+str(obj_class)] = pred_class

    preds = preds.drop('object_id', axis=1)
    print(preds.head())

    score_i = log_loss(list(metadata.target), preds)
    print("Train set result: %.3f" % score_i)


score_train()
