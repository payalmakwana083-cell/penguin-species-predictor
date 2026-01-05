import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
df = pd.read_csv('data/penguins_size.csv')
df = df.dropna()
print(df.columns)
df = df[df['sex'].str.contains('MALE|FEMALE')]

# 3. Encode categorical data
df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})
island_map = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
df['island'] = df['island'].map(island_map)

# 4. Define Features (X)
X = df[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['species']

# 5. Train and Save
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'penguin_model.pkl')

print("Model trained and saved successfully as penguin_model.pkl!")
