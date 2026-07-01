import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.utils import resample

print("=" * 60)
print("GP DNA PREDICTION MODEL TRAINING")
print("=" * 60)

os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)

#  STEP 1: Generate Synthetic Records from NHS Patterns 
print("\nGenerating appointment records from NHS data patterns...")
np.random.seed(42)

# DNA rates from NHS England Appointments in General Practice dataset
# Source: https://digital.nhs.uk/data-and-information/publications/statistical/
#         appointments-in-general-practice
# Contains information from NHS England, licensed under OGL v3.0
nhs_patterns = {
    'GP':                   0.072,
    'Nurse':                0.065,
    'Other Practice staff': 0.080,
    'Mental Health':        0.110,
}
mode_patterns = {
    'Face-to-face': 0.075,
    'Telephone':    0.068,
    'Video/Online': 0.082,
}

appointments = []
n_appointments = 10000

for _ in range(n_appointments):
    hcp_type   = np.random.choice(list(nhs_patterns.keys()),
                                  p=[0.50, 0.30, 0.16, 0.04])
    appt_mode  = np.random.choice(list(mode_patterns.keys()),
                                  p=[0.60, 0.30, 0.10])
    age            = int(np.clip(np.random.normal(45, 20), 0, 100))
    hour           = np.random.randint(8, 18)
    day_of_week    = np.random.randint(0, 5)
    lead_time      = int(np.clip(np.random.exponential(7), 0, 60))

    # Synthetic patient-level features
    # prior_dna_count: number of DNAs in last 12 months (zero-inflated)
    prior_dna_count   = np.random.choice([0, 1, 2, 3, 4, 5],
                                          p=[0.72, 0.14, 0.07, 0.04, 0.02, 0.01])
    # deprivation_score: 1-10 IMD-style score (1=least deprived)
    deprivation_score = int(np.clip(np.random.beta(2, 3) * 10 + 1, 1, 10))

    # Appointment-level modifier (small relative contributions)
    base_modifier = (
        (nhs_patterns[hcp_type] - 0.072)
        + (0.04 if age < 25   else -0.01 if age > 65 else 0)
        + (0.03 if hour < 10  else  0.01 if hour >= 16 else 0)
        + (lead_time / 60) * 0.08
        + (deprivation_score - 5) / 5 * 0.04
        + (mode_patterns[appt_mode] - 0.075)
    )

    # Prior DNA history is the dominant predictor — creates learnable signal
    if prior_dna_count >= 3:
        dna_prob = np.clip(0.42 + base_modifier + np.random.normal(0, 0.03),
                           0.25, 0.65)
    elif prior_dna_count == 2:
        dna_prob = np.clip(0.25 + base_modifier + np.random.normal(0, 0.03),
                           0.12, 0.45)
    elif prior_dna_count == 1:
        dna_prob = np.clip(0.14 + base_modifier + np.random.normal(0, 0.02),
                           0.06, 0.30)
    else:
        dna_prob = np.clip(0.05 + base_modifier + np.random.normal(0, 0.02),
                           0.01, 0.18)

    dna = 1 if np.random.random() < dna_prob else 0

    appointments.append({
        'hcp_type':           hcp_type,
        'appt_mode':          appt_mode,
        'age':                age,
        'hour':               hour,
        'day_of_week':        day_of_week,
        'lead_time':          lead_time,
        'prior_dna_count':    prior_dna_count,
        'deprivation_score':  deprivation_score,
        'dna':                dna
    })

df = pd.DataFrame(appointments)
print(f"Generated {len(df):,} appointments")
print(f"DNA rate: {df['dna'].mean():.2%}  ({df['dna'].sum()} DNAs)")
df.to_csv('../data/nhs_data.csv', index=False)
print("Data saved to ../data/nhs_data.csv")

#  STEP 2: Feature Engineering 
print("\nPreparing features...")
df_encoded   = pd.get_dummies(df, columns=['hcp_type', 'appt_mode'])
feature_cols = [c for c in df_encoded.columns if c != 'dna']
X = df_encoded[feature_cols]
y = df_encoded['dna']
print(f"Features: {len(feature_cols)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Split: {len(X_train):,} train  |  {len(X_test):,} test")

#  STEP 3: Oversample minority class in training set 
# DNA rate ~11% is imbalanced; oversample to 25% positive in training
train_df  = pd.concat([X_train, y_train], axis=1)
majority  = train_df[train_df.dna == 0]
minority  = train_df[train_df.dna == 1]
minority_up = resample(minority, replace=True,
                       n_samples=len(majority) // 3,
                       random_state=42)
train_bal = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
X_train_b = train_bal.drop('dna', axis=1)
y_train_b = train_bal['dna']
print(f"Balanced training: {len(X_train_b):,} rows  |  DNA rate {y_train_b.mean():.2%}")

#  STEP 4: Train Model 
print("\nTraining model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    min_samples_leaf=15,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_b, y_train_b)

# Use probability threshold of 0.35 (tuned for precision/recall balance)
THRESHOLD   = 0.35
proba       = model.predict_proba(X_test)[:, 1]
predictions = (proba >= THRESHOLD).astype(int)
train_pred  = (model.predict_proba(X_train)[:, 1] >= THRESHOLD).astype(int)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test,  predictions)
auc       = roc_auc_score(y_test, proba)

#  STEP 5: Results 
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Training Accuracy : {train_acc:.2%}")
print(f"Test Accuracy     : {test_acc:.2%}")
print(f"ROC-AUC           : {auc:.2f}  (primary metric for imbalanced data)")
print("\nClassification Report:")
print(classification_report(y_test, predictions,
      target_names=['Attended', 'DNA'], zero_division=0))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)
print("=" * 60)

# Save model and features
joblib.dump(model,        '../model/dna_model.pkl')
joblib.dump(feature_cols, '../model/feature_columns.pkl')
joblib.dump(THRESHOLD,    '../model/threshold.pkl')
print("\nModel saved to ../model/dna_model.pkl")
print("Features saved to ../model/feature_columns.pkl")
print("Threshold saved to ../model/threshold.pkl")
print("\nTraining complete!")
