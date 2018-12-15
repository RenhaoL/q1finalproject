import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = 'final-project-data.csv'

df = pd.read_csv(data, names = ['Sampling_Site', 'Water Depth (m)', 'Sand (%)', 'Silt (%)', 'Clay (%)', 'SSA (m2 g−1)', '%TOC', 'N/C', 'δ13C (‰)',
    'T‐ALK (µg/g dw)', 'CPI15–19',    'CPI25–33',    'TAR', '1/Pmar‐aq',   'BIT', 'Λ8(mg/100 mg)',  'C/V', 'S/V', 'LPVI',    '(Ad/Al)V',    '3,5‐Bd (mg/100 mg OC)', '3,5‐Bd/V', 'P/(S + V)'])

features = ['Water Depth (m)', 'Sand (%)', 'Silt (%)', 'Clay (%)', 'SSA (m2 g−1)', '%TOC', 'N/C', 'δ13C (‰)', 'T‐ALK (µg/g dw)', 'CPI15–19',    'CPI25–33',    'TAR', '1/Pmar‐aq',   'BIT', 'Λ8(mg/100 mg)',  'C/V', 'S/V', 'LPVI',    '(Ad/Al)V',    '3,5‐Bd (mg/100 mg OC)', '3,5‐Bd/V', 'P/(S + V)']


x = df.loc[:, features].values

x = StandardScaler().fit_transform(x)


y = df.loc[:,['Sampling_Site']].values

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1 (46.4%)', 'principal component 2 (19.8%)'])


finalDf = pd.concat([principalDf, df[['Sampling_Site']]], axis = 1)


colors = ['blue', 'aqua', 'beige', 'brown', 'coral', 'yellowgreen',
'gold', 'green', 'grey', 'tomato', 'khaki', 'lime',
'navy', 'olive', 'orange', 'orangered', 'pink', 'tan',
'salmon', 'red', 'purple', 'violet']

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1,1,1)

sns.set()
sns.set_style("whitegrid")
sns.set_context("notebook")
ax = sns.scatterplot(x = 'principal component 1 (46.4%)', y = 'principal component 2 (19.8%)', hue =finalDf.Sampling_Site, data = finalDf, palette = colors, markers = False)
plt.suptitle('PCA Result')

plt.show()

print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))
