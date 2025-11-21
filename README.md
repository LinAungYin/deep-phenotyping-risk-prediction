# deep-phenotyping-risk-prediction
A Java implementation of Logistic Regression from scratch for a Computational Deep Phenotyping clinical risk prediction model.

Computational Deep Phenotyping for Clinical Risk Prediction (Java)

This repository contains a self-contained Java implementation of a Logistic Regression model trained via Gradient Descent, showcasing fundamental machine learning and data handling skills within a clinical informatics context.

This project was developed as a technical showcase for the PhD application to the NUS Yong Loo Lin School of Medicine.

**Project Goal**

The core objective is to simulate the use of high-dimensional, derived patient features (Deep Phenotypes) to calculate the probability of a binary clinical event (e.g., High Risk of Readmission, Low Risk of Stroke).

# **1. The Deep Phenotyping Process**

In clinical research, Deep Phenotyping refers to the comprehensive and precise analysis of a patient's observable traits by integrating complex data (like lab results, time-series data, and clinical notes) into structured, meaningful features.

The DeepPhenotypingEngine class simulates this data preparation, generating synthetic patient records based on four key features, which serve as proxies for complex clinical health profiles:

- Age

- Glucose Level (Metabolic health indicator)

- Comorbidity Score (A derived metric of overall disease burden)

- Systolic BP (Cardiovascular health indicator)

# **2. Model Mechanism: Logistic Regression from Scratch**

The core computational work is done in the LogisticRegressionModel class, which implements the classification algorithm without relying on any external machine learning libraries.

## A. Prediction Function

The model works like an advanced, weighted checklist:

It calculates a Raw Score by multiplying each input feature (e.g., Age) by a corresponding, learned Weight (or coefficient).


$$\text{Raw Score} = (\text{Weight}_{\text{Age}} \times \text{Age}) + \dots + \text{Bias}$$

This Raw Score is then passed through the Sigmoid Function (a smooth S-shaped curve). 

 This function converts the raw numerical score (which can range from negative infinity to positive infinity) into a clear Probability value between 0 and 1. This probability is the final clinical risk prediction.

## B. Learning with Gradient Descent

The model finds the optimal weights by minimizing its error using Batch Gradient Descent.

Objective: Find the combination of weights that minimizes the total error (Cost/Loss) between the model's predictions and the true patient outcomes.

Mechanism: Imagine the error as a hilly landscape. The model iteratively takes small steps down the steepest slope (the negative of the gradient) until it reaches the lowest valley (the point of minimum error/highest accuracy).

The Java code iterates 10,000 times, adjusting the weights slightly in the right direction in each step, guaranteeing that the model continually improves its predictive power.

# How to Run the Code

This project requires a standard Java Development Kit (JDK) installed.

Save the file: Save the content of ClinicalRiskPredictionApp.java.

Compile:

<u><b>javac ClinicalRiskPredictionApp.java</b></u>


Run:

<u><b>java ClinicalRiskPredictionApp</b></u>


The output will show the iterative cost reduction during training and the final risk prediction for a simulated new patient.
