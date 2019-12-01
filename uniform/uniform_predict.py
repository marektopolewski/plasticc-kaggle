import pandas as pd
import numpy as np  
import gc                               
def get_pred_columns(include_99):
    gc.enable()
    sample = pd.read_csv('/modules/cs342/Assignment2/sample_submission.csv', nrows=3)
    if include_99: columns = list(sample.columns)[1:len(sample.columns)]
    else: columns = list(sample.columns)[1:len(sample.columns)-1]
    
    del sample
    gc.collect()
    return columns

def inter_galactic_classes():
    gc.enable()
    meta_temp = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
    classes = list(meta_temp[meta_temp.hostgal_photoz==0].target.unique())
    del meta_temp
    gc.collect()
    return classes
    
def extra_galactic_classes():
    gc.enable()
    meta_temp = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
    classes = list(meta_temp[meta_temp.hostgal_photoz!=0].target.unique())
    del meta_temp
    gc.collect()
    return classes
    
def predict_uniform_3(metadata_path):
    gc.enable()
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[['object_id','hostgal_photoz']]
    print('Data loaded')
    inter_classes = inter_galactic_classes()
    extra_classes = extra_galactic_classes()
    all_classes = list(np.unique(inter_classes+extra_classes))
    inter_prob = 1.0/(len(inter_classes)+1)
    extra_prob = 1.0/(len(extra_classes)+1) 
    
    print(inter_classes, len(inter_classes), inter_prob)
    print(extra_classes, len(extra_classes), extra_prob)
    print(all_classes, len(all_classes))
    
    count, max_count = 0.0, float(len(metadata.object_id.unique()))
    preds = pd.DataFrame(metadata.object_id, columns=['object_id'])
    for o_c in all_classes+[99]:
        if o_c == 99:
            pred_class = [inter_prob if h==0 else extra_prob for h in metadata.hostgal_photoz]
        elif o_c in inter_classes:
            print('galacitc %s' % o_c)
            pred_class = [inter_prob if h==0 else 0 for h in metadata.hostgal_photoz]
        else:
            print('extragalacitc %s' % o_c)
            pred_class = [extra_prob if h!=0 else 0 for h in metadata.hostgal_photoz]
        preds['class_'+str(o_c)] = pred_class

    preds.to_csv('predictions_uniform.csv', index=False, header=True)
    print('Predictions exported to "predictions_uniform.csv"')

    print('! Finished !')

def predict_uniform_2(metadata_path):                                       # worse than predict_uniform_1
    gc.enable()
    metadata = pd.read_csv(metadata_path)
    print('Data loaded')
    preds = pd.DataFrame(metadata.object_id, columns=['object_id'])
    for obj_class in get_pred_columns(False):
        preds[obj_class] = float(1)/14
        preds['class_99'] = 0

    print('Predictions made')
    preds.to_csv('predictions_uniform.csv', index=False, header=True, float_format='%.6f')
    print('Predictions exported to "predictions_uniform.csv"')
    
    print('! Finished !')
    
def predict_uniform_1(metadata_path):
    gc.enable()
    metadata = pd.read_csv(metadata_path)
    print('Data loaded')
    preds = pd.DataFrame(metadata.object_id, columns=['object_id'])
    for obj_class in get_pred_columns(False):
        preds[obj_class] = float(1)/15
    
    preds['class_99'] = float(1)/15
    print('Predictions made')
    
    preds.to_csv('predictions_uniform.csv', index=False, header=True, float_format='%.6f')
    print('Predictions exported to "predictions_uniform.csv"')
    
    print('! Finished !')

predict_uniform_3('/modules/cs342/Assignment2/test_set_metadata.csv')
            
    