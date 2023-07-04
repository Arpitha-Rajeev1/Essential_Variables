import matplotlib.pyplot as plt, numpy as np, pandas as pd, mpl_toolkits.mplot3d
from sklearn.preprocessing import StandardScaler   # scales data to standard normal distribution
from sklearn.decomposition import PCA              # principal component analysis
from sklearn.linear_model import LinearRegression # linear regression
from pca import pca

# Import CSV file as a Pandas DataFrame 
heating_data = pd.read_csv('Heating-data.csv', delimiter='\t', index_col='Date') # loads CSV to Pandas

# preprocess them for scikit-learn:
features = ["Sunshine duration [h/day]", "Outdoor temperature [°C]", "Solar yield [kWh/day]", "Solar pump [h/day]", "Valve [h/day]"]

# Separate the target variable 'y' (gas consumption) from the other variables 'x'
target = "Gas consumption [kWh/day]" #dependent variable

x = np.c_[heating_data[features]] # extracts feature values as a matrix
y = np.c_[heating_data[target]] # extracts target values as a one-column matrix

# Scaling the data
model1 = StandardScaler()
model1 = model1.fit(x)
x_scaled = model1.transform(x) # compute and store the scaled data

# Principal component analysis of the feature data
model2 = PCA(2) # two principal components ...
model2 = model2.fit(x_scaled, y)
X_trans = model2.transform(x_scaled) # compute and store the projected data

# Print and plot the principal components as a heat map
print(model2.components_)
plt.figure(figsize=(7,4))
plt.imshow(model2.components_)
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=60, ha="right")
yticks = [""] * len(model2.components_[:,0])
for i in range(len(yticks)):
    yticks[i] = "PC " + str(i+1)
plt.yticks(range(len(yticks)), yticks)
plt.tight_layout()
plt.show()

# Print and plot the principal components as a biplot
model = pca(n_components=2, verbose=0)
results = model.fit_transform(X_scaled, col_labels=features, row_labels=df['Gas consumption [kWh/day]'], verbose=0)
fig, ax = model.biplot(
figsize=(10,5), PC=[0,1], SPE=True, alpha=0.7, cmap="rainbow", color_arrow='black', verbose=0, label=None) 
fig.tight_layout()
