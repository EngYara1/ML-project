import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Original data/Social_Network_Ads.csv')
label = LabelEncoder()
df["Gender"] = label.fit_transform(df["Gender"])

X = df.drop(columns=["Purchased"])
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

features = ['Gender' ,'Age', 'EstimatedSalary']
target = 'Purchased'

scaler = StandardScaler()
scaler.fit(X_train[features])

X_train_scaled = scaler.transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

pd.DataFrame(X_train_scaled, columns=features).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("X_test.csv", index=False)
y_train.to_csv("Y_train.csv", index=False)
y_test.to_csv("Y_test.csv", index=False)

df_vis = df.copy()
df_vis["Purchased"] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_vis, x="Age", y="EstimatedSalary", hue="Purchased", palette="coolwarm")
plt.title("Age vs. Estimated Salary Colored by Purchase Status")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend(title="Purchased")
plt.grid(True)
plt.tight_layout()
plt.show()

models = {
    "DecisionTree": DecisionTreeClassifier(
    random_state=42,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2
),
    "RandomForest": RandomForestClassifier(
    random_state=42,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2
),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=2000, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}

for name, model in models.items():
    print(f"\n== Training {name} ==")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred))