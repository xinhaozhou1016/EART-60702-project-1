# EART-60702-project-1
# ===============================================================
# EART60702 Project 1 – Group 2
# Environmental Science: Temperature, Precipitation and FSNS
# ===============================================================

# ===============================================================
# 1. Import Packages
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)

# ===============================================================
# 2. Load Dataset
# ===============================================================

df = pd.read_csv("project_1.csv")

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Convert time
df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')

# Rename variables
df = df.rename(columns={
    'time': 'Time',
    'TREFHT': 'Temp (K)',
    'PRECT': 'Precip (m/s)',
    'FSNS': 'FSNS (W/m2)'
})

# Keep relevant variables
df = df[['Time', 'Temp (K)', 'Precip (m/s)', 'FSNS (W/m2)']]

# Fix negative precipitation
df.loc[df['Precip (m/s)'] < 0, 'Precip (m/s)'] = 0

print("\nData Overview:")
print(df.describe())

# ===============================================================
# 3. Descriptive Statistics & Visualization
# ===============================================================

# 3.1 Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df['Temp (K)'], kde=True, ax=axes[0], color="#D55E00")
axes[0].set_title("Distribution of Temperature")

sns.histplot(df['Precip (m/s)'], kde=True, ax=axes[1], color="#0072B2")
axes[1].set_title("Distribution of Precipitation")

sns.histplot(df['FSNS (W/m2)'], kde=True, ax=axes[2], color="#6A3D9A")
axes[2].set_title("Distribution of FSNS")

plt.tight_layout()
plt.show()


# 3.2 Boxplots
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['Temp (K)', 'Precip (m/s)', 'FSNS (W/m2)']])
plt.title("Boxplots of Key Variables")
plt.show()


# 3.3 Correlation Heatmap
plt.figure(figsize=(6,5))
corr = df[['Temp (K)', 'Precip (m/s)', 'FSNS (W/m2)']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()


# 3.4 Scatter Relationships
sns.pairplot(df[['Temp (K)', 'Precip (m/s)', 'FSNS (W/m2)']],
             kind="reg",
             diag_kind="kde",
             plot_kws={'line_kws':{'color':'black'}})
plt.show()


# ===============================================================
# 4. Regression Model
# Model: FSNS ~ Temp + Precip + Temp*Precip
# ===============================================================

# Interaction term model
model = smf.ols('Q("FSNS (W/m2)") ~ Q("Temp (K)") * Q("Precip (m/s)")', data=df).fit()

print(model.summary())


# ===============================================================
# 5. Coefficient Plot
# ===============================================================

params = model.params
conf = model.conf_int()
conf['coef'] = params
conf.columns = ['lower', 'upper', 'coef']

plt.figure(figsize=(6,4))
plt.errorbar(conf['coef'], conf.index,
             xerr=[conf['coef'] - conf['lower'], conf['upper'] - conf['coef']],
             fmt='o')
plt.axvline(0, color='red', linestyle='--')
plt.title("Regression Coefficients with 95% CI")
plt.show()


# ===============================================================
# 6. Marginal Effect of Temperature
# ===============================================================

temp_range = np.linspace(df['Temp (K)'].min(), df['Temp (K)'].max(), 100)
mean_precip = df['Precip (m/s)'].mean()

pred_df = pd.DataFrame({
    'Temp (K)': temp_range,
    'Precip (m/s)': mean_precip
})

pred = model.get_prediction(pred_df)
pred_summary = pred.summary_frame()

plt.figure(figsize=(6,4))
plt.plot(temp_range, pred_summary['mean'])
plt.fill_between(temp_range,
                 pred_summary['mean_ci_lower'],
                 pred_summary['mean_ci_upper'],
                 alpha=0.3)
plt.title("Marginal Effect of Temperature on FSNS")
plt.xlabel("Temperature (K)")
plt.ylabel("Predicted FSNS")
plt.show()


# ===============================================================
# 7. Interaction Effect Plot
# ===============================================================

low_precip = df['Precip (m/s)'].quantile(0.25)
mid_precip = df['Precip (m/s)'].quantile(0.5)
high_precip = df['Precip (m/s)'].quantile(0.75)

plt.figure(figsize=(7,5))

for p, label in zip([low_precip, mid_precip, high_precip],
                    ['Low Precip', 'Medium Precip', 'High Precip']):
    
    pred_df = pd.DataFrame({
        'Temp (K)': temp_range,
        'Precip (m/s)': p
    })
    
    pred = model.predict(pred_df)
    plt.plot(temp_range, pred, label=label)

plt.legend()
plt.title("Interaction Effect: Temperature × Precipitation")
plt.xlabel("Temperature (K)")
plt.ylabel("Predicted FSNS")
plt.show()


# ===============================================================
# 8. Model Fit (Predicted vs Actual)
# ===============================================================

df['Predicted FSNS'] = model.predict(df)

plt.figure(figsize=(6,5))
sns.scatterplot(x='FSNS (W/m2)', y='Predicted FSNS', data=df, alpha=0.4)
plt.plot(df['FSNS (W/m2)'], df['FSNS (W/m2)'], color='red')
plt.title("Actual vs Predicted FSNS")
plt.show()


# ===============================================================
# 9. Correlation Tests
# ===============================================================

pearson_corr, p_val = pearsonr(df['Temp (K)'], df['FSNS (W/m2)'])
print("\nPearson correlation (Temp vs FSNS):", pearson_corr)

spearman_corr, sp_p = spearmanr(df['Temp (K)'], df['FSNS (W/m2)'])
print("Spearman correlation (Temp vs FSNS):", spearman_corr)
