import streamlit as st

pages = {
    "My portfolio": [
        st.Page("main_page/main_page.py", title="Main Page"),
        st.Page("survey/finding_friends.py", title="Survey analysis"),
        st.Page("iris/iris.py", title="Iris analysis"),
        st.Page("titanic/titanic.py", title="Titanic analysis"),
        st.Page("marathon/marathon.py", title="Marathon analysis"),
        st.Page("data_predicter/data_predicter_beta.py", title="Data Predicter (beta)")
    ],

}

pg = st.navigation(pages)
pg.run()