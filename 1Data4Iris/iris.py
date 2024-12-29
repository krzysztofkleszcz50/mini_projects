import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import base64

mode = st.sidebar.radio("Choose your mode", ['Introduction', 'Machine learning', 'Data analysis', 'Presentation'])

if mode == 'Introduction':
    
    st.title("LearningApp - Iris! 💐")
    st.write("""
             
    Join me in exploring Iris flowers. 😀
    Discover their distinctive characteristics and understand how machine learning can effectively categorize these.
    \n My application offers:

    - **Model Classification**: Classify your own iris! 
    - **In-Depth Data Analysis**: You can see data analysis in great form.
    - **Various Presentation Options**: Slides, Articles, Notebooks, Email.

    Whether you prefer model classification, data analysis, or creating engaging presentations, this app has all the resources you need. 
    """)



if mode == 'Machine learning':
     
    MODEL_NAME = 'iris_classification_pipeline'

    @st.cache_data
    def load_classification_model():
        return load_model(MODEL_NAME)

    model = load_classification_model()

    def classify_iris(input_data):
        input_df = pd.DataFrame([input_data])
        predictions = predict_model(model, data=input_df)
        return predictions

    st.title("Machine learning 👓")

    tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Values input", "Key feature", "Confusion Matrix"])

    with tab1:
        st.header("Machine learning introduction 💡")
        st.write("""\n In this session, we will focus on training a classification model to differentiate 
                    \n You can even check if this model has right!
                    \n To do this, please see Data analysis section.
                    \n To start this epic journey, please click tab Values input 🌎""")

    with tab2:
        st.header("Please put your values: 👀")
        sepal_length = st.number_input("Długość działki kielicha", min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Szerokość działki kielicha", min_value=0.0, max_value=10.0, value=3.0)
        petal_length = st.number_input("Długość płatka", min_value=0.0, max_value=10.0, value=1.0)
        petal_width = st.number_input("Szerokość płatka", min_value=0.0, max_value=10.0, value=0.5)

        # Data input
        input_data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }

        if st.button("Classify ⚔"):
            result = classify_iris(input_data)
            st.write(f"**This will be:** {result['prediction_label'].values[0]}")

    with tab3:    
        st.title('Key feature 🔑')

        st.image('key_feature.png', caption='key_feauture')

    with tab4:
        st.title('Matrix 💎')

        st.image('1Data4Iris/matrix.png', caption='Matrix')

if mode == 'Data analysis':
     
    st.title("Iris Analysis 📚")

    # Data loading
    df = pd.read_csv('1Data4Iris/25__iris.csv', sep=",")
    df.columns = ('długość kielicha', 'szerokość kielicha', 'długość płatka', 'szerokość płatka', 'klasa')
        
    with st.expander("Analysis summary"):
                    st.write("""
                        This dataset offers insights into the beauty of
                        Iris species, featuring information on three unique varieties.
                        * The data provided for analysis is of very high quality - no missing values and well diversified.
                        * By excluding data transformation, we preserved their original quality.
                        * Sepal length and width are generally larger than petal length and width.
                        * When comparing petal length and width, the data for each group is very diverse, and each group has unique values.""")

    # Switching tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Data Analysis", "Correlations", "Matrix", "Boxplot"])

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

if mode == 'Presentation':
    
    tab1, tab2, tab3, tab4 = st.tabs(["Slides", "Article", "Notebook", "Mail"])

    with tab1:
        st.title('Iris! - Presentation 💻')
        presentation_path = "1Data4Iris/iris_presentation.pdf"

        try:
            with open(presentation_path, "rb") as file:
                notebook_content = file.read()

            st.download_button(
                label="Download iris_presentation.pdf",
                data=notebook_content,
                file_name="iris_slides.pdf",
                mime="application/pdf"
            )
        except FileNotFoundError:
            st.error(f"Plik {presentation_path} nie został znaleziony. Upewnij się, że plik znajduje się w odpowiedniej lokalizacji.")

        with tab2:
            st.title('Iris! - Article 📢')
            presentation_path = "1Data4Iris/iris_pdf.pdf"

            try:
                with open(presentation_path, "rb") as file:
                    notebook_content = file.read()

                st.download_button(
                    label="Download iris_Article.pdf",
                    data=notebook_content,
                    file_name="iris_Article.pdf",
                    mime="application/pdf"
                )
            except FileNotFoundError:
                st.error(f"Plik {presentation_path} nie został znaleziony. Upewnij się, że plik znajduje się w odpowiedniej lokalizacji.")

        with tab3:
             
            st.title('Iris! - Notebook ⛳')

            notebook_path = "1Data4Iris/iris.ipynb"

            with open(notebook_path, 'r', encoding='utf-8') as file:
                notebook_content = file.read()

            st.download_button(
                label="Download iris.ipynb",
                data=notebook_content,
                file_name="iris.ipynb",
                mime="application/octet-stream"
            )

        with tab4:
             
            st.title('Iris! - Mail ✨')
            st.write('''Subject: Summary of Iris Data Analysis

                        Iris Analysis

                        Good morning,
                        Please see below the summary of iris presentation.
                        This dataset offers insights into three unique varieties: Iris setosa, Iris versicolor, and Iris virginica. 
                        Our goal is to explore relationships among these iris classes.

                        General Data Overview 
                        The dataset contains 150 records, with numerical data except for the class column. Data is balanced across classes.
                        There are no missing values in the dataset. Sepal dimensions are generally larger than petal dimensions.

                        Analysis of Relationships 
                        Iris-versicolor and Iris-virginica show similar sepal dimensions, 
                        while Iris-setosa stands out with the widest sepals and shortest petals. 
                        Petal length and width are strongly correlated, unlike sepal dimensions.
                        Iris virginica has the most outliers, followed by Iris-setosa.

                        Analysis Summary
                        High-quality data with no missing values. 
                        Sepal dimensions are larger than petal dimensions, with each iris class displaying unique characteristics.

                        Thank you for your attention.
                        Best regards,
                        Krzysztof Kleszcz

''')



             


