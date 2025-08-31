import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(r"D:\ML Projects\Iris Flower Clasification\Iris.csv")
df.drop(columns=['Id'], inplace=True)

# Train-Test Split
X = df.drop(columns=['Species'])
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Iris Classifier", layout="wide")

st.title("🌸 Iris Flower Classifier App")
st.markdown("This app uses **Random Forest Classifier** to predict the species of Iris flowers.")

# Sidebar - Input Features
st.sidebar.header("Enter Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))

# Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)

st.subheader("🔮 Prediction")
st.write(f"Predicted Species: **{prediction}**")

st.subheader("📊 Prediction Probabilities")
st.bar_chart(pd.DataFrame(proba, columns=model.classes_))

# -----------------------------
# Model Performance
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance on Test Data")
st.write(f"Accuracy: **{accuracy:.2f}**")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
fig, ax = plt.subplots(figsize=(5, 4))  # ✅ create fig before using it
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)