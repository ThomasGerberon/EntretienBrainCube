import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

df = pd.read_csv("C:/Users/tutou/Documents/Professionnel/BrainCubeGit/EntretienBrainCube/depenses_anonymes.csv")

print(df.head())
print(df.shape)

#objet regression lineaire
RegressionSimple = LinearRegression()

y = df["depenses"].values.reshape(-1,1)
X = df["salaire"].values.reshape(-1,1)
RegressionSimple.fit(X,y)
print(RegressionSimple.intercept_)
print(RegressionSimple.coef_)
predict = RegressionSimple.predict(X)
residual = y - predict

'''On trouve comme coefficient a = 0.67949119 et un b = -0.00021333
On a donc une equation de type y= a*salaire + b + erreur
'''

plt.scatter(X,y, label="Vraies valeurs")
plt.plot(X, predict, c='red', label="Regression calculé")
plt.legend()
#plt.plot(residual, "bo")
plt.show()


df['prediction_depense'] = predict

print(df.head())
res_df = df[['salaire', 'depenses', 'prediction_depense']]
print(res_df.head())

#res_df.to_csv("predictions_1.csv", index=False)

#------------------------------------------------------------------------------


RegressionMultiple = LinearRegression()
y = df["depenses"]
X = df[["salaire","age"]]
RegressionMultiple.fit(X,y)
print(RegressionMultiple.intercept_)
print(RegressionMultiple.coef_)
predict2 = RegressionMultiple.predict(X)

'''On a cette fois ci 2 variables explicatives
y = 0.67622724*salaire + 0.13985057*age + -0.0005216905435364922
'''

df["prediction2_depense"] = predict2
print(df.head())

res_df_2 = df[['salaire', 'depenses', 'prediction2_depense']]
print(res_df_2.head())

#res_df_2.to_csv("predictions_2.csv", index=False)

#-------------------------------------------------------------------

# Etape 5 sur le même fichier

df_original = pd.read_csv("C:/Users/tutou/Documents/Professionnel/BrainCubeGit/EntretienBrainCube/depenses.csv")

df_moins_de_50 = df_original.loc[df_original["depenses"] < 50]
print(df_moins_de_50.head())
print(df_moins_de_50.shape)

print(df_moins_de_50['ville'].value_counts())

# On a 68 individus qui ont dépensé moins de 50€, et il y a 28 individus de Clermont-Ferrand

proba_Q1 = 28/68
percentage_Q1 = proba_Q1*100
print("La proba pour qu'un individus ayant dépense moins de 50€ habite a Clermont-Ferrand est de :")
print(proba_Q1, " soit un pourcentage de : ", percentage_Q1,"%")


depense_par_ville = df_original.groupby("ville")["depenses"].sum()
depense_par_ville_salaire_max = df_original.groupby("ville")["salaire"].max()
print(depense_par_ville)
print(depense_par_ville_salaire_max)

print(depense_par_ville/depense_par_ville_salaire_max)
print((depense_par_ville/depense_par_ville_salaire_max).max())


