import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Analiza półmaratonu wrocławskiego 2024')

# Wczytywanie danych
df = pd.read_csv("marathon/halfmarathon_wroclaw_2024__final.csv", sep=";")
df = df.drop(columns=['Nazwisko'])

# Sidebar z filtrami
with st.sidebar:
    name = st.text_input("Imię")
    countries = st.multiselect('Kraj', sorted(df["Kraj"].dropna().unique()), format_func=lambda x: f'🌍 {x}')
    cities = st.multiselect('Miasto', sorted(df["Miasto"].dropna().unique()), format_func=lambda x: f'🏙️ {x}')
    age_categories = st.multiselect('Wiek', sorted(df["Kategoria wiekowa"].dropna().unique()), format_func=lambda x: f'👶 {x}' if 'U' in x else f'👴 {x}')
    gender = st.radio("Płeć", ["Wszyscy", "Mężczyźni", "Kobiety"], format_func=lambda x: f'♂️ {x}' if x == "Mężczyźni" else f'♀️ {x}' if x == "Kobiety" else f'⚧️ {x}')

# Filtrowanie danych
if countries:
    df = df[df["Kraj"].isin(countries)]
if cities:
    df = df[df["Miasto"].isin(cities)]
if age_categories:
    df = df[df["Kategoria wiekowa"].isin(age_categories)]
if name:
    df = df[df["Imię"].str.contains(name, case=False)]
if gender == "Mężczyźni":
    df = df[df["Płeć"] == "M"]
elif gender == "Kobiety":
    df = df[df["Płeć"] == "K"]

# Sprawdzenie, czy DataFrame nie jest pusty
if not df.empty:
    # Liczba zawodników
    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric("Liczba zawodników", df.shape[0])
    with c1:
        st.metric("Liczba mężczyzn", df[df["Płeć"] == "M"].shape[0])
    with c2:
        st.metric("Liczba kobiet", df[df["Płeć"] == "K"].shape[0])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data overview", "Top 5", "Histogram", "Barplot", "Chart"])

    with tab1:
        x = min(10, len(df))
        st.dataframe(df.sample(x).style.format(precision=2), use_container_width=True, hide_index=True)

    with tab2:
        top_columns = ["Miejsce", "Numer startowy", "Imię", "Miasto", "Kraj", "Czas"]
        st.dataframe(df.sort_values("Miejsce")[top_columns].head(5).style.background_gradient(cmap='coolwarm').set_properties(**{'border-color': 'white'}), hide_index=True)

    with tab3:
        # Plotting a histogram of the 'Tempo' column
        fig = plt.figure(figsize=(10, 6))
        plt.hist(df['Tempo'].dropna(), bins=30, color='skyblue')
        plt.title('Distribution of Running Tempo')
        plt.xlabel('Tempo (minutes per kilometer)')
        plt.ylabel('Frequency')
        plt.grid(True)

        st.pyplot(fig)

    with tab4:
        # Clean the data by dropping rows with NaN values in 'Płeć' and 'Tempo'
        df_cleaned = df.dropna(subset=['Płeć', 'Tempo'])

        # Group by gender and calculate average tempo
        average_tempo_by_gender = df_cleaned.groupby('Płeć')['Tempo'].mean()

        # Plot the average tempo by gender
        fig = plt.figure(figsize=(8, 6))
        average_tempo_by_gender.plot(kind='bar', color=['blue', 'pink'])
        plt.title('Average Tempo by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Average Tempo (min/km)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with tab5:
        df_filtered = df[df['Kraj'] != 'POL']

        # Plot a bar chart of the number of participants from each country
        country_counts = df_filtered['Kraj'].value_counts()

        fig2 = plt.figure(figsize=(10, 6))
        country_counts.plot(kind='bar')
        plt.title('Number of Participants by Country (excluding Poland)')
        plt.xlabel('Country')
        plt.ylabel('Number of Participants')
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig2)
else:
    st.write("Brak danych do wyświetlenia po zastosowaniu filtrów.")
