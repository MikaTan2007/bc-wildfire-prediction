# BC Wildfire Prediction Model

Data extraction from ECMWF/ERA5_LAND/DAILY_AGGR using Google Earth Engine (GEE). 

3 features recorded for each point:
- Temperature 2m above ground
- Total Precipitation
- Dew point temperature 2m above ground

Fire Database taken from National Fire Database by the Canadian Wildland Fire Information System (CWFIS)
22619 fires occured in BC between 2010-2024

I generate 22619 random points in BC over a landmasked area between the months of April and October (wildfire season) and years of 2010-2024. Then for each point, I access GEE and extract the three features.

The data is fed into the neural network into a standard train/test/validation split of 70/15/15. We have 3 layers, going from 16 -> 8 -> 1 nodes, applying a ReLU activation function after each layer. The training epoch is 3000.

These are the following results for the prediction model trained on weather features taken from the same day as the wildfire.

```
==================================================
        BC WILDFIRE MODEL CONFUSION MATRIX        
==================================================
                     | Predicted 0 | Predicted 1
                     | (No Fire)   | (Fire)    
--------------------------------------------------
Actual 0 (No Fire)   | 3230        | 31        
--------------------------------------------------
Actual 1 (Fire)      | 129         | 3321      
==================================================

=====================================================
                Classification Report                
=====================================================
              precision    recall  f1-score   support

     No Fire       0.96      0.99      0.98      3261
        Fire       0.99      0.96      0.98      3450
```

To ensure the model is performing without any overfitting on the datasets, I introduce another test on never-seen-before data, i.e, 2025 BC Wildfire information which was not included in the train/test/validation dataset. Following the same pre-processing steps to introduce weather features to the data, I run the 2025 all-fire dataset through the already-trained model, and achieve these results:

```
============================================
      2025 ALL-FIRE DATASET EVALUATION    
============================================
Total Actual Fires Evaluated        : 1378
Model Predicted as 'Fire' (1)       : 1310
Model Predicted as 'No Fire' (0)    : 68
Recall (Accuracy on Fires)          : 95.07%
```

I then decide to train the model on the weather features three days before the wildfire, producing the following results:

```
==================================================
        BC WILDFIRE MODEL CONFUSION MATRIX        
==================================================
                     | Predicted 0 | Predicted 1
                     | (No Fire)   | (Fire)    
--------------------------------------------------
Actual 0 (No Fire)   | 3278        | 49        
--------------------------------------------------
Actual 1 (Fire)      | 164         | 3220      
==================================================
```

```
=====================================================
                Classification Report                
=====================================================
              precision    recall  f1-score   support

     No Fire       0.95      0.99      0.97      3327
        Fire       0.99      0.95      0.97      3384

    accuracy                           0.97      6711
   macro avg       0.97      0.97      0.97      6711
weighted avg       0.97      0.97      0.97      6711
```