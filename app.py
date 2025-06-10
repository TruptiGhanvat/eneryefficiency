import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="SmartEnergy Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("ENB2012_data.xlsx")
    return df

df = load_data()
X = df.iloc[:, :-2]
y1 = df.iloc[:, -2]  # Heating Load
y2 = df.iloc[:, -1]  # Cooling Load

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Go to:", ["View Raw Data", "Summary", "Charts", "Predict"])

# 1. RAW DATA
if option == "View Raw Data":
    st.title("🗂️ Raw Data")
    if st.checkbox("Show Data"):
        st.dataframe(df)

# 2. SUMMARY
elif option == "Summary":
    st.title("📊 Data Summary")
    st.subheader("First 5 rows")
    st.write(df.head())
    st.subheader("Statistical Summary")
    st.write(df.describe())
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# 3. CHARTS
elif option == "Charts":
    st.title("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Pairplot")
        st.write("Showing relationships between variables")
        fig2 = sns.pairplot(df.iloc[:, [0, 1, -2, -1]])  # Select few columns
        st.pyplot(fig2)

# 4. PREDICT
elif option == "Predict":
    st.title("🏠 SmartEnergy Predictor")
    st.write("This app predicts **Heating Load** and **Cooling Load** based on building design inputs.")

    # Train models
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=42)
    model1 = RandomForestRegressor().fit(X_train1, y_train1)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=42)
    model2 = RandomForestRegressor().fit(X_train2, y_train2)

    st.sidebar.header("🔧 Input Parameters")
    input_data = {}
    for col in X.columns:
        value = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data[col] = value

    input_df = pd.DataFrame([input_data])

    # Predictions
    pred1 = model1.predict(input_df)[0]
    pred2 = model2.predict(input_df)[0]

    st.subheader("🧾 Your Inputs")
    st.write(input_df)

    st.subheader("🔍 Predicted Energy Loads")
    st.metric("Heating Load (Y1)", f"{pred1:.2f} kWh/m²")
    st.metric("Cooling Load (Y2)", f"{pred2:.2f} kWh/m²")

    # Optional bar chart
    st.subheader("📊 Prediction Chart")
    chart_df = pd.DataFrame({
        'Load Type': ['Heating Load (Y1)', 'Cooling Load (Y2)'],
        'Predicted Value': [pred1, pred2]
    })
    st.bar_chart(chart_df.set_index('Load Type'))

