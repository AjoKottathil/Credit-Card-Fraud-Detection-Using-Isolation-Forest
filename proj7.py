import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
data = pd.read_csv("creditcard.csv")
X = data.drop("Class", axis=1)
y = data["Class"]
model = IsolationForest(
    contamination=0.002,
    random_state=42
)

model.fit(X)
pred = model.predict(X)

pred = [1 if x==-1 else 0 for x in pred]
print(classification_report(y, pred))
import matplotlib.pyplot as plt

fraud = data['Class'].value_counts()

fraud.plot(kind='bar')

plt.title("Fraud vs Normal Transactions")

plt.show()
