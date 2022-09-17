# Ex02-OUTLIER

# AIM
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
 
# EXPLANATION
   
 An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.  

# ALGORITHM

### STEP 1
Read the given Data.

### STEP 2
Get the information about the data.

### STEP 3
Detect the Outliers using IQR method and Z score.

### STEP 4
Remove the outliers.

### STEP 5
Plot the datas using Box Plot.

# PROGRAM

```
Developed by :  A K MOHAN RAJ
Registration Number : 212221230064
```
```
import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)
```

# OUTPUT

### DATASET FOR BHP_CSV
![1](https://user-images.githubusercontent.com/95160497/190847388-14a240e2-d91e-4577-aadc-ed3750389e88.png)

### DATASET HEAD(BHP)
![2](https://user-images.githubusercontent.com/95160497/190847400-a40fe81e-3cb0-4325-8578-7a2c6f7179b2.png)

### DATASET DESCRIBE(BHP)
![3](https://user-images.githubusercontent.com/95160497/190847413-9aea07ad-8953-408c-b3c5-926f0e308f87.png)

### DATASET INFO(BHP)
![4](https://user-images.githubusercontent.com/95160497/190847429-2ef52a95-18af-4183-af74-380841416989.png)

### DATASET NULL VALUES(BHP)
![5](https://user-images.githubusercontent.com/95160497/190847437-8aec2fb9-62cc-4196-90d5-376eb65ff705.png)

### DATASET SHAPE WITH OUTLIERS(BHP)
![6](https://user-images.githubusercontent.com/95160497/190847506-a67e60cb-7527-423b-851a-767a6e17debc.png)

### DATASET BOXPLOT WITH OUTLIERS(BHP)
![7](https://user-images.githubusercontent.com/95160497/190847538-df12eac0-8a24-4174-bc46-9d30dbe624a3.png)

### DATASET WITHOUT OUTLIERS(BHP)
![8](https://user-images.githubusercontent.com/95160497/190847554-ed63502c-2b0c-48dd-9f07-226ad1916a85.png)
![9](https://user-images.githubusercontent.com/95160497/190847573-eecfb681-a231-4abd-9273-3a7caa505460.png)

### DATASET SHAPE WITHOUT OUTLIERS(BHP)
![10](https://user-images.githubusercontent.com/95160497/190847580-0efa1cbf-daa5-4727-832b-76fbdf9314e8.png)

### DATASET BOXPLOT WITHOUT OUTLIERS(BHP)
![11](https://user-images.githubusercontent.com/95160497/190847586-fef0822e-e282-48ed-9f72-570fb4bc81a2.png)

### DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![12](https://user-images.githubusercontent.com/95160497/190847594-2df6a8df-e602-4f13-9b7f-85390e1f67b4.png)

### DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![13](https://user-images.githubusercontent.com/95160497/190847605-ebe2177f-7756-4693-bb82-bae512b3bb84.png)

### DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![14](https://user-images.githubusercontent.com/95160497/190847629-a66f7740-98c4-462f-a68d-61995b5ebf53.png)

### D![Uploading 14.png…]()
ATASET FOR WEIGHT_HEIGHT_CSV
![15](https://user-images.githubusercontent.com/95160497/190847634-ffc888c8-ab3b-416b-ad54-ce4dfe25b77f.png)

### DATASET HEAD(WEIGHT_HEIGHT)
![16](https://user-images.githubusercontent.com/95160497/190847640-97245bf4-e475-45d5-bdc3-ecf4737785c4.png)

### DATASET INFO(WEIGHT_HEIGHT)
![17](https://user-images.githubusercontent.com/95160497/190847656-deded824-0564-4f6f-88e4-b533e2c452b0.png)

### DATASET DESCRIBE(WEIGHT_HEIGHT)
![18](https://user-images.githubusercontent.com/95160497/190847662-01fd15a5-5ee5-4339-8b31-7fa05feea5c9.png)

### DATASET NULL VALUES(WEIGHT_HEIGHT)
![19](https://user-images.githubusercontent.com/95160497/190847667-ae7aebb1-fcf0-45b6-8309-ec0d54810212.png)

### DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)
![20](https://user-images.githubusercontent.com/95160497/190847678-8ed69cae-1b98-4eef-ac13-af7d8c850085.png)

### DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)
![22](https://user-images.githubusercontent.com/95160497/190847726-f4686bc4-fe6f-4a55-aa5d-435fc60c5be8.png)
![21](https://user-images.githubusercontent.com/95160497/190847731-041ab7c3-7047-434f-ad9a-d7f81736d34d.png)

### DATASET SHAPE(WEIGHT_HEIGHT)
### DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)


# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
