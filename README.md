# DSCI_601

This repository is for the course Applied Data Science 1, for its final project documentation.

# Description

Title: **Recommendation of Refactoring Techniques to address Self-Admitted Technical Debt**

Authors: 
* Abdullah A AlsalehData (aa6304@rit.edu)
* Vinayak Sengupta (vs4016@rit.edu)
* Mohamed Wiem Mkaouer - Project Advisor (mwmvse@rit.edu)

About:

The goal of the project is to support software developers in improving the quality of their code by the recommendation of the appropriate refactoring strategies to address Self-Admitted Technical Debt (SATD). To do so, we are designing and implementing a recommendation model that takes as input of existing SATD comments, and recommends the appropriate refactoring operations that needs to be performed as part of addressing the debt in the comment. Along with that we are also going to be classifying among which SATD comments is refactoring even required.


## Requirements

* Python 3 (3.8)
* numpy (1.18.5)
* pandas (1.0.5)
* pickle ( 0.7.5)
* scikit-learn (0.23.1)
* natural language tool kit (3.5)

## Dataset

The dataset for the project is curated towards the identification of refactoring labels for a given SATD/Non-SATD statement. The main columns of the data are the ‘Class’ and the ‘Text’ columns containing the refactoring labels and the refactoring needing comments, respectively. We will be working with 4 unique refactoring label clases. The dataset of 4009 rows with a unique instance of comments for each corresponding label. Hence, we have 2 learnable parameters to account for.  

## Models used
* Random Forest Classifier
* Logistic Regression
* Support Vector Machine (SVM)
* Multi Nomial Naive Bayes (MNB)



## Results





## How to Run
* Clone the project
* Run the test.py file to see the predicted result based on pickled train models.
