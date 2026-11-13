import pandas as pd
import sklearn 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
Data= pd.read_csv('nhanes_periodontal_history_2017_2020.csv')
dictionary = pd.read_csv("variable_dictionary.csv")
age_col = "age_started_regular_cigarette_smoking" 
Data.replace(5.397605346934028e-79, 0.0, inplace=True)
quit_col = "time_since_quitting_cigarettes_value"
empty_columns = Data.columns[Data.isna().all()].tolist()
Data.drop(columns=empty_columns, inplace=True)
#print(Data.isnull().sum())
#print(Data.select_dtypes(include=['str']).columns)
#print(Data['diabetes_status'])
Data["tried_to_quit_smoking_last_12_months_code"] = (Data["tried_to_quit_smoking_last_12_months_code"].fillna(0))
survey = [ "survey_psu","interview_sample_weight","survey_stratum",'mec_exam_sample_weight']
excluded_features = ["participant_id"]
special_missing_codes = {
    "smoked_100_cigarettes_lifetime_code": [7, 9],
    "dental_professional_discussed_blood_sugar": [9],
    "dental_professional_discussed_tobacco_cessation": [9],
    "reason_for_last_dental_visit_code": [7, 9],
    "last_dental_visit_code": [77, 99],
    "education_code_adults": [7, 9],
    "cigarettes_per_smoking_day_last_30_days": [777, 999],
    "days_smoked_cigarettes_last_30_days": [77, 99],
    "time_since_quitting_cigarettes_value": [99999],
    "age_started_regular_cigarette_smoking": [777, 999],
    "time_since_quitting_cigarettes_value": [77777, 99999]
}
for col , codes in special_missing_codes.items():
    Data[col]= Data[col].replace(codes,np.nan)
    Data["never_smoked_regularly"] = np.select(
    [Data[age_col].isna(), Data[age_col].eq(0)],
    [2, 1],
    default=0
)
Data.loc[Data[age_col] == 0, age_col] = np.nan
Data["quit_50plus_years"] = np.select(
    [Data[quit_col].isna(), Data[quit_col].eq(66666)],
    [2, 1],
    default=0
)
Data.loc[Data[quit_col] == 66666, quit_col] = np.nan
Data.drop(columns=["participant_id"],inplace=True)
Data.dropna(subset=['periodontal_bone_loss_reported'], inplace=True)
x = Data.drop(columns=['periodontal_bone_loss_reported', *survey])
y = Data['periodontal_bone_loss_reported']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train,test_size=0.2,random_state=42,stratify=y_train)
numeric_features = numeric_features = [
    "average_drinks_per_drinking_day_code",
    "cigarettes_per_smoking_day_last_30_days",
    "days_smoked_cigarettes_last_30_days",
    "time_since_quitting_cigarettes_value",
    "age_started_regular_cigarette_smoking",
    "days_used_floss_or_interdental_device_last_week",
    "waist_circumference_cm",
    "bmi_kg_m2",
    "family_income_poverty_ratio",
    "age_years"
]
categorical_features = [
    "heavy_drinking_almost_daily_reported",
    "alcohol_frequency_past_12_months_code",
    "ever_had_at_least_one_alcoholic_drink_code",
    "tried_to_quit_smoking_last_12_months_code",
    "time_since_quitting_cigarettes_unit_code",
    "current_cigarette_use_code_1_daily_2_some_days_3_not_at_all",
    "smoked_100_cigarettes_lifetime_code",
    "self_rated_teeth_and_gum_health_1_excellent_5_poor",
    "oral_health_work_difficulty_frequency_code",
    "mouth_pain_frequency_code_1_often_5_never",
    "dental_professional_discussed_blood_sugar",
    "dental_professional_discussed_tobacco_cessation",
    "reason_for_last_dental_visit_code",
    "last_dental_visit_code",
    "high_cholesterol_reported",
    "hypertension_reported",
    "diabetes_status",
    "gum_disease_self_perceived",
    "gum_disease_treatment_reported",
    "current_cigarette_smoker",
    "smoking_status",
    "education_code_adults",
    "race_ethnicity_code",
    "sex_code_1_male_2_female"
]
numeric_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy='median')),
 ('normalize', StandardScaler())
])
categorical_encoder = Pipeline([
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('encoder',OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
('numeric', numeric_pipeline,numeric_features),
('categorical' , categorical_encoder,categorical_features)
])
model = Pipeline([
('preprocessing', preprocessor),
('RandomForest', RandomForestClassifier(random_state=42,class_weight='balanced_subsample',max_depth=10,max_features='log2',min_samples_leaf=2,n_estimators=200,criterion='entropy'))])
model.fit(x_train_new, y_train_new)
threshold = 0.46
test_probabilities = model.predict_proba(x_test)[:, 1]
predictions = (test_probabilities >= threshold).astype(int)
#probabilities = model.predict_proba(x_val)[:, 1]
# results = []

# for threshold in np.arange(0.05, 0.71, 0.01):

#     predictions_val = (
#         probabilities >= threshold
#     ).astype(int)

#     score = f1_score(
#         y_val,
#         predictions_val,
#         average='macro'
#     )

#     results.append((score, threshold))
# results.sort(key=lambda x: x[0], reverse=True)
# for score, threshold in results:
#     print(
#         f"Macro F1: {score:.4f} | "
#         f"Threshold: {threshold:.2f}"
#     )
# for threshold in [0.46, 0.50]:

#     predictions_val = (probabilities >= threshold).astype(int)

#     score = f1_score(
#         y_val,
#         predictions_val,
#         average='macro'
#     )

#     print(f"Threshold: {threshold} | Macro F1: {score:.4f}")
# model = ImbPipeline([
#     ('preprocessing', preprocessor),
#     ('sampler', 'passthrough'),
#     ('RandomForest', RandomForestClassifier(
#         n_estimators=200,
#         max_depth=10,
#         min_samples_leaf=2,
#         max_features='log2',
#         criterion='entropy',
#         random_state=42
#     ))])
# params = {
#     'RandomForest__max_features': [
#         'sqrt', 'log2', 0.3, 0.5, 0.7, 1.0
#     ],

#     'RandomForest__criterion': [
#         'gini', 'entropy', 'log_loss'
#     ]
# }

# params = [
#     {
#         'sampler': ['passthrough'],
#         'RandomForest__class_weight': [
#             None,
#             'balanced',
#             'balanced_subsample'
#         ]
#     },
#  {
#         'sampler': [RandomOverSampler(random_state=42)],
#         'sampler__sampling_strategy': [
#             0.25, 0.30, 0.40, 0.50,
#             0.60, 0.75, 1.0
#         ],
#         'RandomForest__class_weight': [None]
#     }
# ]
# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# grid = GridSearchCV(
#     model,
#     params,
#     scoring='f1_macro',
#     cv=cv,
#     n_jobs=-1,
#     verbose=1,
#     refit=False
# )

# grid.fit(x_train_new, y_train_new)

# results = sorted(
#     zip(
#         grid.cv_results_['mean_test_score'],
#         grid.cv_results_['params']
#     ),
#     key=lambda item: item[0],
#     reverse=True
# )
# for score, parameters in results:
#     print(
#         f"Macro F1: {score:.4f} | "
#         f"Parameters: {parameters}"
#     )
# model = model.fit(x_train,y_train)
# predictions=model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Macro F1:", f1_score(y_test, predictions, average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
disp=ConfusionMatrixDisplay.from_predictions(y_test,predictions,display_labels=['No', 'Yes'],cmap='Purples',values_format='d')
plt.title('Logistic Confusion Matrix')
macro_f1 = f1_score(y_test, predictions, average='macro')
disp.ax_.set_title(
    f'Logistic Regression Confusion Matrix\n'
    f'Threshold: {threshold:.2f} | '
    f'Accuracy: {accuracy:.2%} | '
    f'Macro F1: {macro_f1:.2%}'
)
plt.tight_layout()
plt.show()

