# Flood Risk Prediction for Road Networks — Kerala 2018

Machine learning model for predicting road segment flood risk during heavy rainfall, developed as an MSc dissertation at the University of Manchester (awarded Distinction).

The study focuses on the 2018 Kerala floods, one of the most severe flood events in India in nearly a century, which caused catastrophic damage to road infrastructure across Ernakulam and Idukki districts.

---

## Overview

Traditional optical satellite imagery is obscured by cloud cover during active flood events, making real-time assessment difficult. This project uses Sentinel-1 SAR imagery, which penetrates cloud cover, to identify flood extent and intersect it with road network data to label flooded segments. A logistic regression classifier is then trained on environmental and topographic features to predict flood likelihood for any road segment given precipitation and terrain conditions.

The goal is to support proactive infrastructure rerouting and emergency response planning ahead of flood events.

---

## Study Area

Ernakulam and Idukki districts, Kerala, India — two of the worst-affected areas during the August 2018 floods.

---

## Dataset

- **68,404 road segments** extracted from OpenStreetMap
- **Flooded segments**: 765 (1.12% of total) — severe class imbalance addressed using SMOTE oversampling
- **11 features** engineered per segment from multiple geospatial sources

| Feature | Source |
|---------|--------|
| Mean / min / max elevation | SRTM DEM |
| Mean / min / max slope | Derived from DEM in ArcGIS Pro |
| Mean / min / max precipitation | CHIRPS (Climate Hazards InfraRed Precipitation with Station data) |
| Imperviousness | Global impervious surface raster (Tsinghua University, via GEE) |
| Distance to drainage | OSM drainage network (Near tool, ArcGIS Pro) |

---

## Methods

### Feature Engineering
- Sentinel-1 SAR GRD imagery (21 August 2018) processed in ESA SNAP using global thresholding to identify water pixels
- Zonal statistics applied to each road segment in ArcGIS Pro to extract feature values and determine flood labelling
- Coordinate system transformations and attribute-level QA performed in ArcGIS Pro

### Class Imbalance
- SMOTE (Synthetic Minority Over-sampling Technique) applied to the training set to balance classes without discarding majority class data

### Feature Selection
- Two methods used in combination: Recursive Feature Elimination (RFE) via scikit-learn, and logistic regression p-value analysis via statsmodels
- RFE identified five significant predictors: mean slope, mean precipitation, imperviousness, min slope, max slope
- Combined with statsmodels results, the final feature set used to train the model was: mean slope, mean precipitation, imperviousness, min slope, max slope, min elevation, and distance to drainage

### Model
- Logistic regression (binary classification: flooded / not flooded)
- 70/30 train-test split; SMOTE applied to training set only; test set uses original unbalanced data
- k-fold cross-validation for generalisation assessment

---

## Results

| Metric | Value |
|--------|-------|
| Overall accuracy | 74% |
| AUC (ROC curve) | 0.74 |
| k-fold cross-validation score | 0.7367 ± 0.0046 |

---

## Files

| File | Description |
|------|-------------|
| `Logistic_regression_classifier_code.ipynb` | Full Jupyter notebook: data preparation, SMOTE, feature selection, model training and validation |
| `flood_risk_dissertation.pdf` | Full MSc dissertation |

---

## Dependencies

```python
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
```

---

## Data Sources

- Sentinel-1 SAR GRD imagery — Copernicus Open Access Hub
- SRTM DEM — NASA / USGS (via OpenTopography)
- Road and drainage network — OpenStreetMap (via Geofabrik)
- Precipitation data — CHIRPS (Funk et al., 2015)
- Imperviousness — Global impervious surface dataset, Tsinghua University (Gong et al., 2020), accessed via Google Earth Engine

---

## Limitations

- Logistic regression assumes a linear relationship between features and flood probability; nonlinear models (Random Forest, XGBoost) may improve performance
- Model trained and validated on a single flood event; generalisation to other events or geographies requires further testing
- Imperviousness and drainage data may not reflect conditions at the time of the 2018 event
- CHIRPS precipitation resolution (~6 km) is coarser than the road segment scale, which limits spatial precision of the precipitation feature
