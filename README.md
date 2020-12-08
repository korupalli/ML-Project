# Retweet Prediction

Requirements: 

scikit-image	0.17.2	
scikit-learn	0.23.1	
scipy	1.5.0	
seaborn	0.10.1	
sklearn	0.0	
statsmodels	0.11.1	
tqdm	4.48.2	
zipp	3.1.0	
Jinja2	2.11.2	
matplotlib	3.2.1	
numpy	1.18.4	
pandas	1.0.3	


Execution Instructions:
- Clone the git repo. It contain train and test sets
- Main dataset is present in google drive since its 2Gb in size or can be taken from this https://competitions.codalab.org/competitions/25276#participate-get_starting_kit download public data
- Run Parent.py which outputs results of Chi-square tests and P-values and Anova f-measure and Kendall coefficient results and runs hypertuning on 3 classifiers prints scores on     test set and classification report for ensemble model
  Preprocessing code is commented in Parent.py file to save time. Uncommenting will run code on main dataset.


Python files information:
HyperTuning.py - Contains gridsearch analysis and ensemble model

Parent.py - Main file which imports from all other files and outputs

Preprocessor.py - Contains code for preprocessing and undersampling

algoSelection.py - Comparion of algorithms

classifiers.py - returns the accuracy and f1_score of each classifier

featureSelection.py - method which are used for feature_selection

vif.py - Contains code for varience inflation factor
