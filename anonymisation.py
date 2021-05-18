import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/tutou/Documents/Professionnel/BrainCubeGit/EntretienBrainCube/depenses.csv")

# Je commence par afficher le dataframe et les dimensions
print(df.shape)
print(df.head())

# Je fais des copies pour faire le test
tmp_df = df.copy()
new_df = df.copy()

def centreeReduit(col, approx=1):
    # Fonction qui fait l'operation centrée reduit
    meanCol = round(df[col].mean(), approx)
    stdCol = round(df[col].std(), approx)
    # formule du theoreme central limite
    tmp_df[col] = (tmp_df[col] - meanCol) / stdCol

def anonyme(col):
    unique_values = df[col].unique()
    res = []
    for i in df[col]:
        index = np.where(unique_values == i)
        index = index[0][0] + 1
        new_value = col + str(index)
        res.append(new_value)
    return res, unique_values


res, liste_ville = anonyme('ville')
print(liste_ville)
print(res)


for i in range(len(tmp_df['nom'])):
    tmp_df.iloc[i,0] = 'id' + str(i+1)


numeric = ['age', 'salaire', 'depenses']

for j in numeric :
    centreeReduit(j)
    #print(tmp_df[j].head())


#plt.hist(tmp_df['age'])
plt.hist(tmp_df['salaire'])
plt.show()

print(tmp_df.head())

new_df['age'] = tmp_df['age']
new_df['salaire'] = tmp_df['salaire']
new_df['depenses'] = tmp_df['depenses']
new_df['ville'] = res
new_df['nom'] = tmp_df['nom']

print("Le dataframe final est créé, il peut mettre mis en fichier csv")
print(new_df.head())

#new_df.to_csv("depenses_anonymes.csv", index=False)




