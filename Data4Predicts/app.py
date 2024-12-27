#Imports

import streamlit as st
from pycaret.classification import setup, compare_models, predict_model, pull
from pycaret.datasets import get_data
import pandas as pd
from faker import Faker
import random

fake = Faker()

# 1. Creating logic and functions of application

# Creating datasets for user (using faker)
column_options = {
    'Name': fake.name,
    'Address': fake.address,
    'Email': fake.email,
    'Experience (years)': lambda: random.normalvariate(10, 2),
    'Job': lambda: random.choice(['Engineer', 'Doctor', 'Teacher', 'Artist', 'Lawyer'])
}

# Function to generate salary based on experience (to make datasets more fit)
def generate_salary(experience):
    base_salary = 30000
    experience_factor = 5000
    return random.normalvariate(base_salary + experience_factor * experience, 10000)

# Function to generate datasets for user
def generate_fake_data(title, rows):
    columns = list(column_options.keys())
    data = {col: [column_options[col]() for _ in range(rows)] for col in columns}
    if 'Experience (years)' in data:
        data['Salary'] = [generate_salary(exp) for exp in data['Experience (years)']]
    df = pd.DataFrame(data)
    df.title = title
    return df

# Function to display datasets
def display_dataset_overview(dataset_name):
    if 'df' in st.session_state and dataset_name == st.session_state.df.title:
        df = st.session_state.df
        st.write(f"Overview of {df.title} dataset 👀")
        st.write("Sample Data 👓")
        st.write(df.head(10))
        st.write("Data Description 👓")
        st.write(df.describe())
        return df
    else:
        dataset = get_data(dataset_name)
        st.write(f"Overview of the {dataset_name} dataset 👀")
        st.write("Sample Data 👓")
        st.write(dataset.head(10))
        st.write("Data Description 👓")
        st.write(dataset.describe())
        return dataset

# Function to display model training parameters
def display_model_parameters(dataset):
    # Only for categorical columns as target
    valid_columns = [col for col in dataset.columns if dataset[col].dtype in [object, 'category']]
    if not valid_columns:
        st.error("No valid categorical columns for target found.")
        return None
    target = st.selectbox('Select Target Feature 🚗', valid_columns)
    return target

# Function to train models
def train_models(dataset, target):
    if target not in dataset.columns:
        st.error("Error: Invalid target feature. Please select a valid target feature.")
        return None
    
    # Convert target column to categorical type if it's not already
    if dataset[target].dtype not in [object, 'category']:
        dataset[target] = dataset[target].astype('category')
    
    # Print out a sample of the dataset and target unique values for verification
    print("Dataset sample:\n", dataset.head())
    print("Target unique values:", dataset[target].unique())

    try:
        setup(data=dataset, target=target, session_id=123)
        best_models = compare_models(n_select=5)
        return best_models
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Function to display best models summary
def display_models_summary(best_models):
    st.write("Model Champions 🎉")
    results = pull()
    st.dataframe(results)
    model_options = [f"Model {i+1}: {str(model).split('(')[0]}" for i, model in enumerate(best_models)]
    if model_options:
        selected_model_name = st.selectbox("Select Your Champion ⚔", model_options)
        selected_model_index = model_options.index(selected_model_name)
        return best_models[selected_model_index], selected_model_name
    else:
        st.warning("No models available. Please check your target feature and try again.")
        return None, None

# Function to display data entry form
def display_data_entry_form(dataset, target):
    st.write("Enter Data for Prediction 🎢")
    form = st.form(key='prediction_form')
    input_data = {}
    with form:
        for col in dataset.columns:
            if col != target:
                input_data[col] = st.text_input(f"{col}", key=col)
        submit_button = st.form_submit_button(label='Submit Data ✔')
    return input_data, submit_button

# Function to predict based on input data
def make_prediction(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = predict_model(model, data=input_df)
    predicted_class = prediction["prediction_label"].values[0]
    return predicted_class

# 2. Creating Main Application

# Creating layout - Introduction

st.title("Train your own model 💡")

with st.expander("Introduction 🌎"):
            ("""Are you ready to unlock the power of machine learning? ✌
            Our app helps you generate custom datasets and train powerful models using PyCaret. 🎉
            Simply select or generate your dataset, choose your target feature, and let us do the heavy lifting. 🛠
            """)

with st.expander("Generate your own data!📚"): 
        st.write("""Are you ready to create your own datasets? 😎
            Imagine yourself that you are the boss and run your own company,
            according this data
            """)
        with st.form(key='generate_data_form'):
            title = st.text_input('Title for the dataset')
            rows = st.number_input('Number of rows', min_value=60, value=60)
            generate_button = st.form_submit_button(label='Add this data')
            if generate_button:
                st.session_state.df = generate_fake_data(title, rows)
                st.session_state.custom_data_generated = True
                st.session_state.custom_data_title = title
                st.success('Data added! ✨')

# Dataset Overview - layout + logic

if 'custom_data_generated' in st.session_state and st.session_state.custom_data_generated:
    dataset_name = st.selectbox('Select Dataset 🧺', ['iris', 'wine', 'titanic', 'juice', st.session_state.custom_data_title])
else:
    dataset_name = st.selectbox('Select Dataset 🧺', ['iris', 'wine', 'titanic', 'juice'])
if st.button("Select Dataset 🧭"):
    st.session_state.dataset = display_dataset_overview(dataset_name)

# Step 2: Choose Model Parameters
if 'dataset' in st.session_state:
    target = display_model_parameters(st.session_state.dataset)
    st.session_state.target = target

# Train Model
if 'target' in st.session_state and st.session_state.target is not None:
    if st.button("Train Model 🚚"):
        st.session_state.train_model_clicked = True
        with st.spinner("Please wait while the model is being trained..."):
            st.session_state.best_models = train_models(st.session_state.dataset, st.session_state.target)
        if st.session_state.best_models is not None:
            st.success("Model training completed!")

# See Trained Model
if 'best_models' in st.session_state and st.session_state.best_models is not None:
    model, model_name = display_models_summary(st.session_state.best_models)
    st.session_state.selected_model = model
    st.session_state.selected_model_name = model_name

# Enter Data for Prediction
if 'selected_model' in st.session_state:
    input_data, submit_button = display_data_entry_form(st.session_state.dataset, st.session_state.target)
    if submit_button:
        st.session_state.input_data = input_data

# Check Prediction
if 'input_data' in st.session_state:
    if st.button("Check Prediction 🔎"):
        st.session_state.check_prediction_clicked = True
        predicted_class = make_prediction(st.session_state.selected_model, st.session_state.input_data)
        st.write(f"#### Predicted Class: {predicted_class}")

# Refresh Data Button
if st.button("Double click to refresh 🌧"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Thank you for take a look on this 😎