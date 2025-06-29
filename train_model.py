#Importing libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#Load dataset
df = pd.read_csv("StudentsPerformance.csv")

#Add target column 'result'
df['average'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['result'] = df['average'].apply(lambda x: 'pass' if x >= 60 else 'fail')
df.drop(columns='average', inplace=True)

#Define features and labels
X = df.drop(columns='result')
y = df['result']

#Identify column types
numeric_cols = ['math score', 'reading score', 'writing score']
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

#Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols)
])

#Final pipeline with model
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier())
])

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
pipeline.fit(X_train, y_train)

#Evaluate
y_pred = pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

#Save model
joblib.dump(pipeline, "student_model.pkl")
print("✅ Model saved as student_model.pkl")
