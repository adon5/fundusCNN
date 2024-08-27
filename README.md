# How to Use:  
(Optional) Download CUDA is not already installed: CUDA 12.1+  
Download the repo: git clone the repo (or equivalent)
Install requirements: e.g. python -m pip install -r requirements.txt (install pytorch 2.3.1 or later, see requirements.txt) 
Upload dataset to a new directory called 'data' with subdirectories containing image files for each class (configured to support four classes: cataract, diabetic_retinopathy, glaucoma, normal).   
Three uses: Training (a new set of model weights to be saved as a pth file), evaluation (performance metrics and a confusion matrix), inference.  

# UI, UX, etc still to be completed.  

# About Performance Metrics:

Accuracy (% correct):  
(TP + TN) / (TP + TN + FP + FN)  

Optimise sensitivity (recall) for screening scenarios (rule in disease)
Why? Increasing TP rate and decreasing FN rate most important as prior probability is low. False positives not considered in this equation.
Recall (Sensitivity):  
TP / (TP + FN)

Precision (PPV):  
TP / (TP + FP)

Optimise specificity for diagnostic scenarios (rule out disease)
Why? As prior probability is relatively high, decreasing the rate of false positives and maximising true negatives are the goal. False negatives not considered in this equation.
Specificity:   
TN / (TN + FP)

F1-Score:  
2 * (Precision * Recall) / (Precision + Recall)
