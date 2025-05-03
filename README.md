# Parkinson's Disease Progression Prediction

This repository contains the implementation of several linear regression models to predict the progression of Parkinson's Disease based on vocal features extracted from patients' voice recordings. The goal is to tailor treatment to individual patients by estimating their total Unified Parkinson's Disease Rating Scale (UPDRS) scores on a daily basis.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction

You can read the full version of report in ```Regression on Parkinson's Disease Data.pdf```. Parkinson's Disease is a neurodegenerative disorder that affects mainly dopamine-producing neurons in the substantia nigra, a region of the brain responsible for dopamine production. Symptoms generally develop slowly over the years, and the progression of symptoms varies from person to person. This project aims to predict the progression of Parkinson's Disease using voice recordings of patients, which can be easily collected through smartphones.

## Dataset

The dataset used in this study is publicly available [here](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring). It contains voice recordings of Parkinson's Disease patients used to generate vocal features that can be used to regress total UPDRS scores. Load the data in the same directory of the repository.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

The required packages are:

numpy,
matplotlib,
seaborn,
pandas,
scipy

## Usage
To run the script, use the following command:

```bash
python main.py
```

This script will load the dataset, preprocess it, and train several linear regression models based on different optimization approaches. The performance of each model will be evaluated and compared.

## Conclusion

Understanding Parkinson's Disease and its progression is the first step to living well with it. The results of this study show that the Local Steepest Descent method is the most reasonable and reliable among the other algorithms for predicting the progression of Parkinson's Disease. The results also suggest that there is a strong correlation between motor UPDRS and total UPDRS scores, and that additional data acquisition methods, such as image processing and mobile application-based solutions, could improve the prediction accuracy of the models.

## References

[1] "Parkinson's Disease Information Page", National Institute of Neurological Disorders and Stroke, https://www.ninds.nih.gov/Disorders/All-Disorders/Parkinsons-Disease-Information-Page

[2] "Parkinson's Telemonitoring Dataset", UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
