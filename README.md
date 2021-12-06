# CS596Fall2021FinalProject
Zhaopu Chen, Donghao Feng, Zunqi Huang, John Lin, Twinkle Lok, Xinrui Lyu, Siyi Song


## 1. Introduction:

Alzheimer’s disease, the most common form of dementia, has challenged human beings for century.  
An early detection would be a great strategy for patients to slow or prevent disease progression.  
Our group would like to use deep learning approaches to help detect for early stage of Alzheimer.

## 2. Data Set:

- Human brain MRI images
- Main differences between patients and healthy individuals: brain volume
- Souce: Alzheimer's Disease Neuroimaging Initiative (ADNI) from USC's Mark and Mary Stevens Neuroimaging and Informatics Institute 

## 3. Cognitive normal vs Alzheimer’s disease:

<img src="Sample1.png" alt="" style="width:45%"> <img src="Sample2.png" alt="" style="width:45%">

## 4. Model:

1. Pytorch wirh Pydicom package
2. CNN
3. IDE: Jupyter

## 5. Expected Results:

Train the ML model successfully and the estimated success rate of images recognition between normal human brain and Alzheimer’s disease is at least 80%.

## 6. Implementation:

1. Data cleaning:
    The raw data is divided into two parts: training data set and testing data set. Each part of the data consists of patient's data and normal person's data. For multiple MRI images of the same person, we clean up data and only take dozens of pictures in the middle, so that the difference between CN's pictures and AD's pictures is greater.
2. Train ML model
3. Verify the model

## 6. Result Display:

Example.
