import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

# Kompanija savo klientams išsiuntinėjo laiškus.
# Žmonės arba pasielgė taip, kaip kompanija norėjo (paspaudė ant nuorodos, nusipirko produktą ir t.t.), arba ne.

df = pd.read_csv('Social_Network_Ads.csv')
print(df)
print(df.describe())
#matome amžiaus pasiskirstymą duomenyse.
sns.displot(df['Age'])
plt.show()
#matome lyties pasiskirstymą duomenyse.
sns.displot(df['Gender'])
plt.show()
#matome lyčių pasiskirstymą duomenyse pagal pirkimus.
sns.violinplot(x='Purchased', y='Gender', data=df)
plt.show()
#matome pirkimų pasiskirstymą duomenyse pagal lytį.
df.plot.scatter('Age', 'Purchased')
plt.show()
#matome pirkimų pasiskirstymą duomenyse pagal pajamas.
df.plot.scatter('EstimatedSalary', 'Purchased')
plt.show()

#duomenų patikrinimas ir sutvarkymas
print(df.isnull().sum())
print(df['Gender'].unique()) # patikriname ar nėra kitų reikšmių nei Male ar Female.

#duomenų pavertimas į float reikšmes.
ohe = OneHotEncoder()
ohe.fit_transform(df.Gender.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
df['male'] = (ohe.fit_transform(df.Gender.values.reshape(-1, 1)).toarray())
print(df)

#duomenų apmokymas ir testavimas.
X = df[['Age', 'male']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Apmokymui panaudojam mokymui skirtus duomenis (70%) t.y. X_train, y_train
model = LogisticRegression()
model.fit(X_train, y_train)
# Spėjimui panaudojom testavimui skirtus duomenis (30%) t.y. X_test
y_predicted = model.predict(X_test)
print(y_predicted)
# Patikrinimui panaudojom testavimui skirtus duomenis (30%) t.y. X_test, y_test
model.score(X_test, y_test)

#Klaidų patikrinimas
print(metrics.confusion_matrix(y_test, y_predicted))
#false positive - I tipo klaida. sergi, nors is tikro nesergi.
#false negative - II antro tipo klaida. nesergi, nors is tikro sergi.
