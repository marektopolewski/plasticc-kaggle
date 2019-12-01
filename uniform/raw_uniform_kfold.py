import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

# score on train data for uniform distro
def score_train():

    metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')

    preds = pd.DataFrame(metadata.object_id, columns=["object_id"])
    obj_classes = sorted(metadata.target.unique())
    # obj_classes = get_class_labels(obj_classes)

    # same proabability of belonging to each class
    for obj_class in obj_classes:
        preds[obj_class] = 1.0/len(obj_classes)
    

    score_i = log_loss(metadata.target, preds.drop('object_id', axis=1))
    print("Train set result: %.3f" % score_i)


score_train()
