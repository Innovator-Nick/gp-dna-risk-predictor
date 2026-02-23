import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

print("=" * 60)
print("🤖 GP DNA PREDICTION MODEL TRAINING")
print("=" * 60)

# ===== STEP 1: Generate Realistic Individual Records =====
print("\n📊 Creating realistic appointment records from NHS patterns...")

np.random.seed(42)

# Real NHS DNA rates by category
nhs_patterns = {
    'GP': 0.072,
    'Nurse': 0.065,
    'Other Practice staff': 0.080,
    'Mental Health': 0.110,
}

mode_patterns = {
    'Face-to-face': 0.075,
    'Telephone': 0.068,
    'Video/Online': 0.082,
}

appointments = []
n_appointments = 10000

for i in range(n_appointments):
    # Random selections
    hcp_type = np.random.choice(list(nhs_patterns.keys()),
                                p=[0.50, 0.30, 0.16, 0.04])
    appt_mode = np.random.choice(list(mode_patterns.keys()),
                                 p=[0.60, 0.30, 0.10])

    age = int(np.clip(np.random.normal(45, 20), 0, 100))
    hour = np.random.randint(8, 18)
    day_of_week = np.random.randint(0, 5)
    lead_time = int(np.clip(np.random.exponential(7), 0, 60))

    # Calculate DNA probability
    base_rate = nhs_patterns[hcp_type]
    mode_factor = mode_patterns[appt_mode] / 0.075
    age_factor = 1.3 if age < 30 else (0.9 if age > 65 else 1.0)
    time_factor = 1.2 if hour < 10 else 1.0
    lead_factor = 1.1 if lead_time > 14 else 1.0

    dna_prob = min(0.25, base_rate * mode_factor * age_factor * time_factor * lead_factor)
    dna = 1 if np.random.random() < dna_prob else 0

    appointments.append({
        'hcp_type': hcp_type,
        'appt_mode': appt_mode,
        'age': age,
        'hour': hour,
        'day_of_week': day_of_week,
        'lead_time': lead_time,
        'dna': dna
    })

df = pd.DataFrame(appointments)
print(f"✅ Generated {len(df):,} appointments")
print(f"📈 DNA rate: {df['dna'].mean():.2%}")

# Save data
df.to_csv('../data/nhs_data.csv', index=False)
print("✅ Data saved to ../data/nhs_data.csv")

# ===== STEP 2: Train Model =====
print("\n⚙️ Preparing features...")

df_encoded = pd.get_dummies(df, columns=['hcp_type', 'appt_mode'])
feature_cols = [col for col in df_encoded.columns if col != 'dna']

X = df_encoded[feature_cols]
y = df_encoded['dna']

print(f"✅ Features: {len(feature_cols)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Split: {len(X_train):,} train, {len(X_test):,} test")

# Train
print("\n🎯 Training model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n" + "=" * 60)
print("📊 MODEL PERFORMANCE")
print("=" * 60)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print("=" * 60)

# Save
joblib.dump(model, '../model/dna_model.pkl')
joblib.dump(feature_cols, '../model/feature_columns.pkl')

print("\n✅ Model saved to ../model/dna_model.pkl")
print("✅ Features saved to ../model/feature_columns.pkl")
print("\n🎉 Training complete!")