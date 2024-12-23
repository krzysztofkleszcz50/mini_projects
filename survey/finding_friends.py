import json
import streamlit as st
import pandas as pd 
from pycaret.clustering import load_model, predict_model 
import plotly.express as px  

MODEL_NAME = 'survey/welcome_survey_clustering_pipeline_v2'

DATA = 'survey/welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'survey/welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header("Witaj przyjacielu! Pomożemy znaleźć Ci fajnych ziomków")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

with st.popover("Imię"):
    st.markdown("Dla lepszej personalizacji")
    name = st.text_input("Możesz podać imię")
st.write("Cześć", name, ", dziękujemy za korzystanie z mojej apki. Pomożemy Ci znaleźć przyjaciół.")

st.header(f"Gratulacje! Jesteś w prestiżowej grupie {predicted_cluster_data['name']}")

with st.expander("Czy masz odwagę dowiedzieć się paru słów o Twoich znajomych?"):
        st.markdown(predicted_cluster_data['description'])
        same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
        st.metric("Liczba twoich znajomych", len(same_cluster_df))

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Świeżak czy doświadczony?", "Jak sobie cenisz naukę?", "Psiarz czy Kociarz?", "Ulubione miejsce?", "Rozkład płci"])

with tab0:
    st.header("Osoby z grupy")
    fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
        title="Rozkład wieku w grupie",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab1:
    fig = px.histogram(same_cluster_df, x="edu_level")
    fig.update_layout(
        title="Rozkład wykształcenia w grupie",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab2:
    fig = px.histogram(same_cluster_df, x="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab3:
    fig = px.histogram(same_cluster_df, x="fav_place")
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab4:
    fig = px.histogram(same_cluster_df, x="gender")
    fig.update_layout(
        title="Rozkład płci w grupie",
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)
