# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:24900982
RegisterNumber:SHEIK FAIZAL S
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/user-attachments/assets/848cb08a-f800-4d4c-866c-64acb30d9f2a)

![Screenshot 2024-12-26 094409](https://github.com/user-attachments/assets/2630af59-d0ee-45a6-88fe-70fc66fc1edb)

![image](https://github.com/user-attachments/assets/c70b3359-290c-4dd9-b8fa-95e83c8157c3)

![image](https://github.com/user-attachments/assets/a9e41803-fcc0-4089-a781-ec2606b3b1c8)

![image](https://github.com/user-attachments/assets/4508f5db-deb9-4ac6-acf5-d901b2165063)

![image](https://github.com/user-attachments/assets/3a00875c-0bc9-4ef3-b0b3-c10bebbc0dca)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
