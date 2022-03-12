import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

df = pd.read_csv('Social_Network_Ads.csv')
print(df)

#duomenų pavertimas į float reikšmes.
df['male'] = (df['Gender'] == 'Male').astype('int') # True "iš principo" = 1, o False "iš principo" = 0
print(df)

# Nusistatome kintamuosius
X = df[['Age','EstimatedSalary', 'male']]
Y = df['Purchased']

# naudojame StandardScaler norėdami apdoroti duomenis dėl tikslumo.
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Naudojame cross validation, kuris skirtas apmokyti duomenis dalimis.
k_fold = KFold(n_splits=5)

test_scores = []
for train_idx , test_idx in k_fold.split(X):
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    model = LogisticRegression()
    model.fit(Xtrain , Ytrain)

    test_scores.append(model.score(Xtest, Ytest))

plt.plot(test_scores)
plt.plot([np.mean(test_scores)]*len(test_scores))
plt.show()

# Nusistatome test scores
print(" Cross validation score : ", np.mean(test_scores))
print(test_scores)