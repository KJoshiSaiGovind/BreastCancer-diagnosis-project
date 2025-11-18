import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Page Setup
st.set_page_config(page_title="Cancer Diagnosis AI", layout="wide")
st.title("🏥 Breast Cancer Diagnosis AI")
st.write("This app uses Random Forest to detect cancer cells based on your data.")

# 2. Load & Clean Data (Cached for speed)
@st.cache_data
def load_and_clean_data():
    # Load
    df = pd.read_csv('data.csv')
    
    # Clean: Force columns to numbers, turn errors (like 'Volume 23') into 0
    # We skip col 0 (ID) and col 1 (Diagnosis) for the features
    X_raw = df.iloc[:, 2:32]
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce')
    X_raw = X_raw.fillna(0)
    
    X = X_raw.values
    
    # Target
    y = df.iloc[:, 1].values
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y.astype(str))
    
    return X, y, df

try:
    with st.spinner("Loading and cleaning data..."):
        X, y, raw_df = load_and_clean_data()
        st.success("Data loaded successfully!")

    # 3. Sidebar Controls
    st.sidebar.header("Settings")
    split_size = st.sidebar.slider("Test Data Size", 0.1, 0.5, 0.25)
    n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 100, 10)

    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # 5. Show Results
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.caption("(Top-Left: True Benign, Bottom-Right: True Malignant)")

    with col2:
        if st.checkbox("Show Raw Data Sample"):
            st.dataframe(raw_df.head(10))

except FileNotFoundError:
    st.error("CRITICAL ERROR: 'data.csv' was not found. Please upload it to GitHub.")
except Exception as e:
    st.error(f"An error occurred: {e}")