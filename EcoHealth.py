import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# CSV dosyasını okuma
df = pd.read_csv("water_pollution_disease.csv")

# Sütun adlarını düzenleme
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

# Eksik verileri doldurma
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Korelasyon Matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()

# Kirlilik ve Hastalıklar Arası İlişki
sns.pairplot(df[["contaminant_level_ppm", "nitrate_level_mg/l", "lead_concentration_µg/l", "diarrhea_rate", "cholera_rate", "typhoid_rate"]])
plt.show()

# Yıllara Göre İshal Vakaları
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="diarrhea_rate", hue="country")
plt.title("Yıllara Göre İshal Vakaları")
plt.show()

# Özellik Seçimi ve Model Hazırlığı
features = ["contaminant_level_ppm", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg/l", "nitrate_level_mg/l", "lead_concentration_µg/l"]
target = "diarrhea_rate"

X = df[features]
y = df[target]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model değerlendirmesi
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

# K-means Kümeleme
numerical_columns = ["contaminant_level_ppm", "ph_level", "turbidity_ntu", "dissolved_oxygen_mg/l", "nitrate_level_mg/l", "lead_concentration_µg/l"]
df_cleaned = df[numerical_columns].dropna()

kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['cluster'] = kmeans.fit_predict(df_cleaned)

# Kümeleme görselleştirmesi
plt.figure(figsize=(10, 6))
sns.scatterplot(x="contaminant_level_ppm", y="nitrate_level_mg/l", hue="cluster", data=df_cleaned, palette="viridis", s=100)
plt.title("K-means Clustering: Su Kirliliği Kümeleri")
plt.xlabel("Contaminant Level (ppm)")
plt.ylabel("Nitrate Level (mg/L)")
plt.show()

# Küme merkezlerinin görselleştirilmesi
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("Kümelerin Merkezleri")
plt.show()
