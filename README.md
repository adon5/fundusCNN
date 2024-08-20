# How to Use:  
(Optional) Download CUDA is not already installed: CUDA 12.1+  
Download the repo: git clone the repo  
Install requirements: python3 -m pip install -r requirements.txt  
Upload dataset to a new directory called 'data' with subdirectories containing image files for each class (configured to support four classes: cataract, diabetic_retinopathy, glaucoma, normal).   
Three uses: Training (a new set of model weights to be saved as a pth file), evaluation (performance metrics and a confusion matrix), inference.  

# UI, UX, etc still to be completed. Early stages, draft, pre-release. Mostly boilerplate stuff atm.    

# About Performance Metrics:

Accuracy (% correct):  
(TP + TN) / (TP + TN + FP + FN)  

Optimise sensitivity (recall) for diagnostic scenarios i.e. when a disease is suspected based on prior clinical findings.  
Why? Increasing TP rate and decreasing FN rate most important as prior probability is high. False positives not considered in this equation.
Recall (Sensitivity):  
TP / (TP + FN)

Precision (PPV):  
TP / (TP + FP)

Optimise specificity for screening-type scenarios i.e. if this algorithm was widely applied to a low-prevalence disease.  
Why? As prior probability is relatively low, decreasing the rate of false positives and maximising true negatives are the goal. False negatives not considered in this equation.
Specificity:   
TN / (TN + FP)

F1-Score:  
2 * (Precision * Recall) / (Precision + Recall)
