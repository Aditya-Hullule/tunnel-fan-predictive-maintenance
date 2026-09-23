import pandas as pd
import time
import os
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


DATASET = "features.csv"

df = pd.read_csv(DATASET)

X = df.drop(columns=["file", "label"])
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


for name, model in models.items():

    print(f"\nTraining {name}...")

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start


    start = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start


    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )


    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )


    model_file = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, model_file)

    model_size_kb = os.path.getsize(model_file) / 1024


    results.append({
        "Model": name,
        "Test Accuracy (%)": accuracy * 100,
        "Macro F1": macro_f1,
        "CV Mean (%)": cv_scores.mean() * 100,
        "CV Std (%)": cv_scores.std() * 100,
        "Training Time (s)": training_time,
        "Prediction Time (ms)": prediction_time * 1000,
        "Model Size (KB)": model_size_kb
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Mean (%)",
    ascending=False
)


print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


results_df.to_csv(
    "model_comparison.csv",
    index=False
)


print("\nResults saved as: model_comparison.csv")