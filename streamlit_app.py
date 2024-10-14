import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


st.title("Project Happiness")

app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])
df = pd.read_csv("whr2023.csv")


if app_page == "Data Exploration":
    st.dataframe(df.head(5))
    st.subheader("01. Description of the dataset")
    st.dataframe(df.describe())
    st.subheader("02. Missing values")
    dfnull = df.isnull()/len(df)*100
    total_missing = (dfnull.sum()).round(2)
    st.write(total_missing)
    st.write(dfnull)

    if total_missing[0] == 0.0:
            st.success("Congrats you have now missing values")

if app_page == "Visualization":
    st.subheader("03. Data Visualization")

    list_columns = df.columns

    values = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Healthy life expectancy"])

    #Creation of the line chart
    st.line_chart(df, x=values[0], y=values[1])


    #Creation of the bar chart
    st.bar_chart(df, x=values[0], y=values[1])


    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables: ", list_columns, ["Healthy life expectancy", "Happiness score", "Logged GDP per capita", "Freedom to make life choices"])

    df2 = df[[values_pairplot[0], values_pairplot[1], values_pairplot[2], values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)


if app_page == "Prediction":
    st.title("03. Prediction")
        
    list_columns = df.columns

    input_lr = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Logged GDP per capita"])

    df_new = df.dropna() # initially I was just droping the null val in the y and not in x so it was giving me valueerror
    df2 = df_new[input_lr]

    X = df2
    y = df_new["Healthy life expectancy"]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    lr.fit(X_train, y_train)

    #Step 6 Evaluation
    mae = metrics.mean_absolute_error(pred, y_test) # how far are we from the truth
    r2 = metrics.r2_score(pred, y_test) # percentage how well you are improving

    st.write("Mean Absolute Error:", mae)
    st.write("R2:", r2)



