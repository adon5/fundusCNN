
# Performance Metrics:

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
