Created by Zhou Quan Eric Gao 
# Problem Overview
The science of training machines to learn and produce models for future predictions is widely used, and not for nothing. Agriculture plays a critical role in the global economy. With the continuing expansion of the human population understanding worldwide crop yield is central to addressing food security challenges and reducing the impacts of climate change.

Crop yield prediction is an important agricultural problem. The Agricultural yield primarily depends on weather conditions (rain, temperature, etc), and pesticides, and accurate information about the  history of crop yield is an important thing for making decisions related to agricultural risk management and future predictions.

This repository was created as the basis research for the Tu Bishvat presentation given to find out the best performing algorithm for crop yield prediction; a specific prediction was tested in a specific scenario for an interactive activity, where
Country: Australia 
Item: Wheat
Year: 2013 
Average Rainfall (mm): 534.0 
Pesticides used per tonne: 45177.18
Average Temperature (°C): 17.4
Area Yield (hg/ha): Unknown

## Data Availability 

The original dataset can be found here: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
All datasets (publicly available datasets) here are taken from FAO (Food and Agriculture Organization) and World Data Bank.
http://www.fao.org/home/en/
https://data.worldbank.org/

## Models Included
This repository contains various machine-learning models for predicting crop yields based on environmental and agricultural factors.

- Linear Regression
- CNN (Convolutional Neural Network) *Not Included in Model Comparison
- Decision Tree
- Random Forest
- KNN (K-Nearest Neighbors)
- SVM (Support Vector Machine)
- Boosting

# Project Structure
## Under the *Figures* Folder:
`info.py`: produces the graphs `Log_Transformed_yield,png` and `hgha_yield.png`
`Visualization.py`: produces the graphs `area_countplot.png`, `item_countplot.png`, `yield_per_country.png`, and `yield_per_crop.png` 

## Under the *Algorithms* Folder:
`Boosting.py` Gradient Boosting Tree 
`CNN.py` Convolutional Neural Network
`DecisionTree.py` Decision Tree 
`KNN.py` K Nearest Neighbor 
`LinearReg.py` Linear Regression 
`RandomForest.py` Random Forest 
`SVM.py` Single Vector Machine

## Results
`Result.py` Evaluates the result of all the machine learning algorithm under three metrics

r^2 (Coefficient of determination), `r^2.png`: the proportion of variance in crop yield that can be explained by the independent variable(s)

Prediciton Specific |Error|, `ScenarioSpecific_Pred.png`: absolute value difference between the predicted yield by the machine learning algorithm and the actual yield

Theoretical, `Theoretical.png`: Theoretical limits the machine learning algorithms can reach with optimal tuning and allowed maximal run time

Overall, based on the three metrics, the ranking for the algorithms for this specific study is as follows:
1. K Nearest Neighbor
2. Random Forest
3. Decision Tree
4. Gradient Boosting Tree
5. Linear Regression 
6. Single Vector Machine
