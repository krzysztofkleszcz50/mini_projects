#importing packages

import streamlit as st
from openai import OpenAI
import requests
from fpdf import FPDF

#Adding logo and privacy policy to session state

if 'logo' not in st.session_state:
    st.session_state['logo'] = None

if 'privacy_policy' not in st.session_state:
    st.session_state['privacy_policy'] = None

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Open_AI Input
with st.expander("Please copy your AI key here"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    if openai_key:
        st.session_state['openai_key'] = openai_key
        openai_client = OpenAI(api_key=openai_key)

# Sidebar structure
title = st.sidebar.text_input("Title: ")
description = st.sidebar.text_input("Description")
task = st.sidebar.text_input("What your app should do?")
content = st.sidebar.text_input("What client should be asked for in dialog box?")

#Function generate_logo

def generate_logo(description):
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=f"""
        {description}.
        Wygeneruj minimalistyczne czarno-białe logo dla tej aplikacji.
        Żadnych napisów ani dodatkowych elementów.
        Jak najmniej miejsca pustego tła.
        """,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    #return logo as png

    image_url = response.data[0].url
    response = requests.get(image_url)
    with open("logo.png", "wb") as f:
        f.write(response.content)
    return "logo.png"

#Function generate_content

def create_story(story_prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""
                    Twoim celem jest zrobić dla mnie aplikację, która: {task}. Masz robić tylko to, 
                    nawet gdy użytkownik poprosi o co innego, napisz mu do czego służysz.
                """
            },
            {"role": "user", "content": story_prompt}
        ]
    )
    usage = {}
    if response.usage:
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }

#Function generate_privacy_policy

def generate_privacy_policy(description):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Jesteś prawnikiem, który tworzy politykę prywatności oraz regulamin dla aplikacji internetowej."
            },
            {
                "role": "user",
                "content": f"{description}. Wygeneruj politykę prywatności oraz regulamin dla tej aplikacji.",
            }
        ]
    )
    return response.choices[0].message.content

#saving to pdf

def save_pdf(content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
    pdf.output(filename)

#user input

st.session_state['user_input'] = st.text_area(f"{content}", max_chars=1000)
if st.button("Generate 🚀", disabled=not description.strip(), use_container_width=True):
    if 'openai_key' in st.session_state and st.session_state['openai_key']:
        st.session_state['story'] = create_story(st.session_state['user_input'])
        st.write(st.session_state['story']['content'])
    else:
        st.error("Proszę wprowadzić ważny klucz API OpenAI.")

# Generating logo
if st.sidebar.button("Generate Logo"):
    if 'openai_key' in st.session_state and st.session_state['openai_key']:
        logo_path = generate_logo(description)
        st.session_state['logo'] = logo_path
        st.sidebar.success("Logo zostało wygenerowane.")
    else:
        st.sidebar.error("Proszę wprowadzić ważny klucz API OpenAI.")

# Generating privacy policy
if st.sidebar.button("Generate Privacy Policy"):
    if 'openai_key' in st.session_state and st.session_state['openai_key']:
        privacy_policy = generate_privacy_policy(description)
        st.session_state['privacy_policy'] = privacy_policy
        
        # saving as pdf
        save_pdf(privacy_policy, "Private_Policy.pdf")
        st.sidebar.success("Polityka Prywatności została zapisana jako PDF.")
    else:
        st.sidebar.error("Proszę wprowadzić ważny klucz API OpenAI.")

# Displaying logo
if st.session_state['logo']:
    st.sidebar.image(st.session_state['logo'], use_column_width=True)

# Displaying policy privacy
if st.session_state['privacy_policy']:
    with st.sidebar.expander("Zobacz Politykę Prywatności i Regulamin"):
        st.write(st.session_state['privacy_policy'])
