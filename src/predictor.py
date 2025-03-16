import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('dataset/2023-2024.csv')

# Create a target variable (1 for win, 0 for loss/draw)
df['Result'] = df.apply(lambda row: 1 if row['GF'] > row['GA'] else 0, axis=1)

# Encode categorical variables (Team and Opponent)
label_encoder = LabelEncoder()
df['Team_encoded'] = label_encoder.fit_transform(df['Team'])
df['Opponent_encoded'] = label_encoder.transform(df['Opponent'])

# Select features and target
features = [
    'Team_encoded', 'Opponent_encoded', 'xG', 'xGA', 'Poss', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt'
]
X = df[features]
y = df['Result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate average statistics for each team
team_stats = df.groupby('Team')[['xG', 'xGA', 'Poss', 'Sh', 'SoT', 'Dist', 'FK', 'PK', 'PKatt']].mean().reset_index()

# Create a DataFrame of unique matches (Team vs. Opponent)
unique_matches = df[['Team', 'Opponent']].drop_duplicates().reset_index(drop=True)

# Merge Team stats into unique_matches
unique_matches = unique_matches.merge(team_stats, how='left', left_on='Team', right_on='Team')
unique_matches = unique_matches.rename(columns={
    'xG': 'Team_xG',
    'xGA': 'Team_xGA',
    'Poss': 'Team_Poss',
    'Sh': 'Team_Sh',
    'SoT': 'Team_SoT',
    'Dist': 'Team_Dist',
    'FK': 'Team_FK',
    'PK': 'Team_PK',
    'PKatt': 'Team_PKatt'
})

# Merge Opponent stats into unique_matches
unique_matches = unique_matches.merge(team_stats, how='left', left_on='Opponent', right_on='Team')
unique_matches = unique_matches.rename(columns={
    'xG': 'Opponent_xG',
    'xGA': 'Opponent_xGA',
    'Poss': 'Opponent_Poss',
    'Sh': 'Opponent_Sh',
    'SoT': 'Opponent_SoT',
    'Dist': 'Opponent_Dist',
    'FK': 'Opponent_FK',
    'PK': 'Opponent_PK',
    'PKatt': 'Opponent_PKatt'
})

# Drop the extra 'Team' column from the second merge
unique_matches = unique_matches.drop(columns=['Team_y'])

# Ensure the 'Team' column is correctly referenced
unique_matches = unique_matches.rename(columns={'Team_x': 'Team'})

# Encode Team and Opponent in unique_matches
unique_matches['Team_encoded'] = label_encoder.transform(unique_matches['Team'])
unique_matches['Opponent_encoded'] = label_encoder.transform(unique_matches['Opponent'])

# Prepare features for prediction
X_unique = unique_matches[[
    'Team_encoded', 'Opponent_encoded',
    'Team_xG', 'Team_xGA', 'Team_Poss', 'Team_Sh', 'Team_SoT', 'Team_Dist', 'Team_FK', 'Team_PK', 'Team_PKatt'
]]

# Rename columns to match the training data
X_unique = X_unique.rename(columns={
    'Team_xG': 'xG', 'Team_xGA': 'xGA', 'Team_Poss': 'Poss', 'Team_Sh': 'Sh', 
    'Team_SoT': 'SoT', 'Team_Dist': 'Dist', 'Team_FK': 'FK', 'Team_PK': 'PK', 'Team_PKatt': 'PKatt'
})

# Ensure column order matches training features
X_unique = X_unique[features]

# Predict win rates
unique_matches['PredictedWinRate'] = model.predict_proba(X_unique)[:, 1]

# Format PredictedWinRate as percentage with two decimal places
unique_matches['PredictedWinRate'] = (unique_matches['PredictedWinRate'] * 100).round(2)

# Save predicted win rates to CSV
output_file = "dataset/match_win_rates.csv"
unique_matches[['Team', 'Opponent', 'PredictedWinRate']].to_csv(output_file, index=False)

print(f"Predicted win rates saved to {output_file}")
