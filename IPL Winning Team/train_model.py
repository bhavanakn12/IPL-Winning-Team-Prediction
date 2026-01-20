import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load datasets
data = pd.read_csv(r"C:\Users\Bhavan KN\OneDrive\Documents\IPL Winning Team\ipl_colab.csv")
match_results = pd.read_csv(r"C:\Users\Bhavan KN\OneDrive\Documents\IPL Winning Team\match_results.csv")

# Sanity check: Print team names
print("Unique batting teams:", data['batting_team'].unique())
print("Unique winner teams:", match_results['winner'].unique())

# Merge results with data
data = data.merge(match_results[['mid', 'winner']], on='mid', how='left')

# Create target: 1 if batting_team is the winner, else 0
data['target'] = (data['batting_team'] == data['winner']).astype(int)

# Check correctness
print("Unique target values:", data['target'].unique())
print("Value counts:\n", data['target'].value_counts())
print(data[['mid', 'batting_team', 'winner', 'target']].head(10))

# If you see only [0] here, fix your match_results.csv as explained above.

# Feature engineering
data['runs_left'] = data['total'] - data['runs']
data['balls_left'] = 120 - (data['overs'] * 6)
data['wickets_left'] = 10 - data['wickets']
data['crr'] = data['runs'] / data['overs'].replace(0, 1)
data['rrr'] = (data['runs_left'] * 6) / data['balls_left'].replace(0, 1)

categorical_features = ['batting_team', 'bowling_team']
numerical_features = ['runs_left', 'balls_left', 'wickets_left', 'crr', 'rrr']

X = data[categorical_features + numerical_features]
y = data['target']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Classes in y_train:", y_train.unique())
print("Counts in y_train:\n", y_train.value_counts())

# Train only if there are at least two classes
if len(y_train.unique()) < 2:
    print("ERROR: Your target variable contains only one class. Fix your match_results.csv so winners match actual teams.")
else:
    model.fit(X_train, y_train)
    pickle.dump(model, open('ipl_win_model.pkl', 'wb'))
    print("Model trained and saved as ipl_win_model.pkl")
