import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

MODEL_NAME_3 = 'titanic/titanic_classification_pipeline_v2'

@st.cache_data
def load_classification_model():
    return load_model(MODEL_NAME_3)

model = load_classification_model()

def classify_survived(input_data):
    # Data transformacion
    input_data['Sex'] = 1 if input_data['Sex'] == 'male' else 0
    input_data['Embarked'] = 0 if input_data['Embarked'] == 'C' else (1 if input_data['Embarked'] == 'Q' else 2)
    
    input_df = pd.DataFrame([input_data])
    predictions = predict_model(model, data=input_df)
    return predictions

# Sidebar 
st.sidebar.header("Wprowadź dane pasażera:")
passenger_class = st.sidebar.selectbox("Klasa pasażera", [1, 2, 3])
sex = st.sidebar.selectbox("Płeć", ["male", "female"])
age = st.sidebar.number_input("Wiek", min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input("Liczba rodzeństwa / małżonków na pokładzie", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Liczba rodziców / dzieci na pokładzie", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Cena biletu", min_value=0.0, max_value=1000.0, value=50.0)
embarked = st.sidebar.selectbox("Port zaokrętowania", ["C", "Q", "S"])

# Input
input_data = {
    'Pclass': passenger_class,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

if st.sidebar.button("Sprawdź szanse przeżycia"):
    result = classify_survived(input_data)
    st.sidebar.write(f"**Szansa na przeżycie:** {result['prediction_label'].values[0]}")

# Main page
st.title("Titanic Analysis")

with st.expander("Introduction 💡"):
    st.write("""
    * Welcome to the Titanic Survival Prediction App!
    * This app helps you predict the chances of survival of a passenger on the Titanic based on various factors such as:
    * Enter the passenger details in the sidebar and click on "Sprawdź szanse przeżycia" to see the prediction. 
    * Let's explore the data and see what factors affected the survival rates. 🔎
    """)

with st.expander("Summary"):
    st.write("""
                * The data turned out to be somewhat inconvenient for analysis due to many missing values.
                * This led to the necessity of data transformation by calculating averages or transforming columns.
                * We can observe that nearly every woman in first and second class survived.
                * The highest chances of survival were for those who embarked from port C.
                * We can observe that the more expensive the ticket, the higher the chance of survival.
                * The most outliers were found in the first class, specifically regarding ticket prices.
             """)

# Ładowanie danych i zmiana nazw kolumn
df = pd.read_csv('titanic/titanic.csv')
df.columns = ['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest']

# Usunięcie kolumn tekstowych, które mogą powodować błędy
df = df.drop(columns=['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])

# Konwersja danych tekstowych na numeryczne
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['embarked'] = df['embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))

# Przełączanie między zakładkami
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ogólny przegląd danych", "Analiza wartości pojedynczych", "Korelacja", "Analiza wartości odstających", "Macierz"])

with tab1:
    st.write(df.sample(10))

with tab2:
    columns = ['pclass', 'age', 'fare', 'parch']
    for col in columns:
        fig = plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(f'{col} (cm)')
        plt.ylabel('Frequency')
        st.pyplot(fig)


with tab3:
    matrix = df.corr()
    colormap = sns.light_palette("blue", as_cmap=True)
    matrix_colored = matrix.style.background_gradient(cmap=colormap)
    st.dataframe(matrix_colored)
    st.subheader("Scatterplot: Fare vs Age")
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='fare', y='age', hue='survived', data=df)
    st.pyplot(plt.gcf())
    
with tab4:
    columns = ['age', 'fare']
    for col in columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, y=col, ax=ax, palette='Set2')
        ax.set_title(f'Boxplot dla kolumny {col}', fontsize=16)
        ax.set_ylabel(f'{col}', fontsize=14)
        st.pyplot(fig)

with tab5:
    survival_rates = df.groupby(['pclass', 'embarked'])['survived'].mean().reset_index()
    pivot_table = survival_rates.pivot(index='pclass', columns='embarked', values='survived')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Survibval Rate'})
    plt.title('Survival Rates by Class and Gender')
    plt.xlabel('Embarked')
    plt.ylabel('Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt.gcf())

    survival_rates = df.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
    pivot_table = survival_rates.pivot(index='pclass', columns='sex', values='survived')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Survival Rate'})
    plt.title('Survival Rates by Class and Gender')
    plt.xlabel('Gender')
    plt.ylabel('Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt.gcf())
