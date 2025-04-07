import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Görselleştirmeler için varsayılan ayarlar
sns.set(style='whitegrid', palette='muted', font_scale=1.1)

# CSV dosyasını okuma
df = pd.read_csv("water_pollution_disease.csv")

# İlk birkaç satıra göz atalım
df.head()

