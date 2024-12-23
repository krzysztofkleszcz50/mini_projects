import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

MODEL_NAME = 'iris/iris_classification_pipeline'

@st.cache_data
def load_classification_model():
    return load_model(MODEL_NAME)

model = load_classification_model()

def classify_iris(input_data):
    input_df = pd.DataFrame([input_data])
    predictions = predict_model(model, data=input_df)
    return predictions

# Sidebar 
st.sidebar.header("Wprowadź parametry irysów:")
sepal_length = st.sidebar.number_input("Długość działki kielicha", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.sidebar.number_input("Szerokość działki kielicha", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.sidebar.number_input("Długość płatka", min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.sidebar.number_input("Szerokość płatka", min_value=0.0, max_value=10.0, value=0.5)

# Data input
input_data = {
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width
}

if st.sidebar.button("Klasyfikuj"):
    result = classify_iris(input_data)
    st.sidebar.write(f"**Przewidywana klasa irysa:** {result['prediction_label'].values[0]}")

# Main page
st.title("Iris Analysis")

# Data loading
df = pd.read_csv('iris/25__iris.csv', sep=",")
df.columns = ('długość kielicha', 'szerokość kielicha', 'długość płatka', 'szerokość płatka', 'klasa')

with st.expander("Introiduction 💡"):
    st.write("""
                * Hi, nice to see you here! Let's dive into the world of irises!
                * Discover the beauty of Iris species through this dataset 🌸
                * Each row represents an individual flower, with measurement values provided in centimeters. In the following study, we will attempt to 
                * identify relationships between the different classes of irises. Are you ready? Let's begin the analysis! 
            """)
    
with st.expander("Analysis summary"):
                 st.write("""
                    * The data provided for analysis is of very high quality - no missing values and well diversified.
                    * By excluding data transformation, we preserved their original quality.
                    * Sepal length and width are generally larger than petal length and width.
                    * When comparing petal length and width, the data for each group is very diverse, and each group has unique values.""")

# Switching tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ogólny przegląd danych", "Analiza wartości pojedynczych", "Korelacje", "Macierz", "Analiza wartości odstających"])

with tab1:
    st.write(df.sample(10))

with tab2:
    columns = ['długość płatka', 'szerokość płatka', 'długość kielicha', 'szerokość kielicha']
    for col in columns:
        fig = plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'{col}')
        plt.xlabel(f'{col} (cm)')
        plt.ylabel('Frequency')
        st.pyplot(fig)

with tab3:    
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='szerokość kielicha', y='długość kielicha', hue='klasa', data=df)
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(10, 6))
    scatter2 = sns.scatterplot(x='szerokość płatka', y='długość płatka', hue='klasa', data=df)
    st.pyplot(plt.gcf())

with tab4:
    matrix = df[["długość kielicha", "szerokość kielicha", "długość płatka", "szerokość płatka"]].corr()
    colormap = sns.light_palette("green", as_cmap=True)
    matrix_colored = matrix.style.background_gradient(cmap=colormap)
    st.dataframe(matrix_colored)

with tab5:
    columns = ['długość płatka', 'szerokość płatka', 'długość kielicha', 'szerokość kielicha']
    for col in columns:
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.boxplot(data=df, x='klasa', y=col, ax=ax, palette='Set2')
        ax.set_title(f'{col}', fontsize=20)
        ax.set_xlabel('Class', fontsize=16)
        ax.set_ylabel(f'{col} (cm)', fontsize=16)
        st.pyplot(fig)


