import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt
import matplotlib.pyplot as plt


st.title("World Happiness Dashboard😁")

app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'Prediction', 'Conclusion'])
df = pd.read_csv("whr2023.csv")



if app_page == "Overview":
    image_path = Image.open("happiness-joy.jpg")
    st.image(image_path, width=400)
    
    st.subheader('Questions we aim to answer: ')
    st.write("What factors most or least affect our happiness?")
    st.write("How are life expectancy and happiness correlated?")
    st.write("How is happiness correlated with income?")
    st.write("How does the country or region where a person lives affect their happiness?")
    st.subheader("Let's explore the dataset!")
    
    st.write("The dataset we will be analyzing is entitled 'World Happiness Report'.")
    st.write("The World Happiness Report reports data from 2023. A preview of the dataset is shown below: ")
    st.dataframe(df.head())
    
    st.write("Information about the dataframe: ")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    
    st.write("Statistics on the dataset: ")
    st.dataframe(df.describe())
    
    st.subheader("Missing values")
    dfnull = df.isnull()/len(df)*100
    total_missing = (dfnull.sum()).round(2)
    st.write(total_missing)
    st.write(dfnull)

    if total_missing[0] == 0.0:
            st.success("Congrats you have now missing values")
    
    st.subheader("Our Goals")
    st.write("The goals of our project are to analyze the factors which effect a person's happiness, and to discover how one's happiness can affect their life expectancy.")
    st.write("Source: https://www.kaggle.com/datasets/atom1991/world-happiness-report-2023 ")
   

if app_page == "Visualization":
    st.subheader("Data Visualization")
    list_columns = df.columns

    values = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Healthy life expectancy"])
    #Creation of the line chart
    st.line_chart(df, x=values[0], y=values[1])


    #Creation of the bar chart
    #st.bar_chart(df, x=values[0], y=values[1])

    string_columns = list(df.select_dtypes(include=['object']).columns)
    data = df.drop(columns=string_columns)
    #.drop(columns=...)

    # Create heatmap data (assuming 'value_column' is the column you want to visualize)
    heatmap_data = data.corr()
    # Calculate correlation matrix

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size as needed
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", ax=ax, cmap='RdBu_r', vmin=-1, vmax=1) 
    plt.xlabel("X-axis Label", fontsize=12)
    plt.ylabel("Y-axis Label", fontsize=12)
    plt.title("Correlation Matrix", fontsize=14)
    st.pyplot(fig)

    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables: ", list_columns, ["Healthy life expectancy", "Happiness score", "Logged GDP per capita", "Freedom to make life choices"])

    df2 = df[[values_pairplot[0], values_pairplot[1], values_pairplot[2], values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)

    st.subheader('Report:')

    #profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    #st_profile_report(profile)

    st.write('Map of Global Happiness for 2023, based off of World Happiness Report Dataset 2023, courtesy of Visual Capitalist.')
    image2 = Image.open('worlds-happiest-countries-2023-MAIN.jpg')
    st.image(image2, width=400)

    st.write('Map of global GDP for 2022, courtesy of World Bank: ')
    image3 = Image.open('capita-worldbank.jpg')
    st.image(image3, width=400)


if app_page == 'Prediction':
    st.title("Prediction")
        
    list_columns = df.columns

    input_lr = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Logged GDP per capita"])

    df_new = df.dropna() 
    df2 = df_new[input_lr]

    X = df2
    y = df_new["Healthy life expectancy"]

    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25 ")
    col1.write(X.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    explained_variance = np.round(mt.explained_variance_score(y_test, pred) * 100, 2)
    mae = np.round(mt.mean_absolute_error(y_test, pred), 2)
    mse = np.round(mt.mean_squared_error(y_test, pred), 2)
    r_square = np.round(mt.r2_score(y_test, pred), 2)

    # Create a comparison DataFrame to visualize Actual vs Predicted values
    comparison_df = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': pred})

    # Display the first 10 rows of the comparison DataFrame
    st.write("### Comparison of Actual vs. Predicted Values")
    st.write(comparison_df.head(10))
    
    # Display results
    st.subheader('🎯 Results')
    st.write("1) The model explains,", explained_variance, "% variance of the target feature")
    st.write("2) The Mean Absolute Error of the model is:", mae)
    st.write("3) MSE: ", mse)
    st.write("4) The R-Square score of the model is", r_square)

    


    # Plotting the Linear Regression line
    st.subheader('📈 Linear Regression Line')

    # Create a scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_new[input_lr[0]], y=y, data=df_new, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    plt.title(f'Linear Regression of Healthy Life Expectancy vs {input_lr[0]}')
    plt.xlabel(input_lr[0])
    plt.ylabel('Healthy Life Expectancy')
    
    st.pyplot(plt)  # Display the plot in Streamlit


if app_page == 'Conclusion':

    st.title('Conclusion')
    st.balloons()

    st.subheader('1. Insights:')
    st.markdown('- Correlation Analysis: Our heatmap and pairplot analyses revealed significant correlations between happiness and other variables. Notably, factors such as healthy life expectancy and GDP per capita had strong positive correlations with happiness scores, which confirms the importance of both wealth and healthiness in increased happiness.')
    st.markdown('- Life Expectancy and Happiness: The linear regression model also demonstrated that healthy life expectancy could be predicted with a reasonable degree of accuracy using features like GDP per capita and happiness score. This shows a strong link between how long people live and their overall well-being.')
    st.subheader('2. Model Performance: ')
    st.markdown("- Our predictive model for healthy life expectancy, using a simple linear regression model, achieved an explained variance score of approximately 72%. This suggests that while our model captures a substantial portion of the variability in life expectancy, there is room for improvement in the model's performance.")
    st.markdown("- The Mean Absolute Error (MAE) and Mean Squared Error (MSE) values were acceptable, indicating that the model predictions were close to the actual values. However, future improvements could focus on enhancing the model by incorporating additional features or using more complex models.")
    st.subheader('3. Ways to Improve Model: ')
    st.markdown("- Data Quality and Feature Engineering: While the current dataset provides a comprehensive look at happiness factors, there are still missing values in some areas. Handling these more effectively, potentially by using imputation strategies or by introducing new variables like mental health or social safety nets, could enhance the model's predictive abilities.")
    st.subheader('4. Longterm Considerations: ')
    st.markdown("- Dynamic Updates: Happiness and well-being are dynamic, influenced by changes in political, economic, and environmental conditions. Continuously updating the model with more recent data, such as future World Happiness Reports, would ensure the model remains relevant and accurate.")
    st.markdown("- Integrating Additional Dataset: To further enhance the depth of analysis, integrating extra datasets, such as those on mental health, education, or environmental factors, could provide new perspectives on what drives happiness around the world.")
