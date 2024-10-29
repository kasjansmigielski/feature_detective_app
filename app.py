import streamlit as st
from pycaret.datasets import get_data
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, finalize_model as reg_finalize_model, predict_model as reg_predict_model, plot_model as reg_plot_model, save_model as reg_save_model
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, finalize_model as cls_finalize_model, predict_model as cls_predict_model, plot_model as cls_plot_model, save_model as cls_save_model
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import dotenv_values, load_dotenv
#from openai import OpenAI
import base64
from langfuse.openai import OpenAI
from langfuse.decorators import observe
import instructor
from pydantic import BaseModel
import json
from typing import List, Dict #, Optional
import time
import random
import boto3
import io
from io import BytesIO
import fsspec
import s3fs
import tempfile

# ustawienia strony w aplikacji
st.set_page_config(layout='wide')

# deklarowanie zmiennych środowiskowych
env = dotenv_values('.env')
load_dotenv()

# konfiguracja pobierania z Digital Ocean Spaces
BUCKET_NAME = 'the-most-important-variables'
FOLDER_NAME = 'datasets/'

# muzyka dla umilenia korzystania z aplikacji :)
MUSIC_FOLDER_NAME = 'music/'

# zapisywane wykresy
CLS_PLOT_FOLDER_NAME = 'plots/classification/'
# REG_PLOT_FOLDER_NAME = 'plots/regression/'
# PLOT_NAME = 'Feature Importance.png'

songs = {
    'Armin van Buuren & David Guetta feat. Aldae - In The Dark' : 'Armin van Buuren & David Guetta feat. Aldae - In The Dark.mp3',
    'Gigi DAgostino - Silence' : 'Gigi DAgostino - Silence.mp3',
    'Titanic Remix Laback feat. Alexis.Carlier - My Heart Will Go On' : 'Titanic Remix Laback feat. @Alexis.Carlier - My Heart Will Go On.mp3'
}

# komunikacja z Cloud Storage
session = boto3.session.Session()
client = session.client('s3',
                        # region_name='fra1', 
                        # endpoint_url='https://fra1.digitaloceanspaces.com',
                        # aws_access_key_id=,
                        # aws_secret_access_key=,
)

# wczytywanie muzyki z Cloud Storage    
def get_mp3_file(bucket_name, folder_name, file_name):
    # przekazywanie pełnej ścieżki do pliku
    file_key = f'{folder_name}{file_name}'
    # pobieranie pliku MP3 jako obiektu
    response = client.get_object(Bucket=bucket_name, Key=file_key)
    # oczytanie danych jako strumień bajtów
    file_data = response['Body'].read()
    return file_data 

# ochrona klucza OpenAI API
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env['OPENAI_API_KEY']
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# dodanie cennika modeli -> per token
model_pricings = {
    'gpt-4o': {
        'input_tokens': 5.00 / 1_000_000,
        'output_tokens': 15.00 / 1_000_000,
    },
    'gpt-4o-mini': {
        'input_tokens': 0.150 / 1_000_000,
        'output_tokens': 0.6 / 1_000_000,
    }
}

# dodanie listy z modelami i załączenie jej do listy cennika
DEFAULT_MODEL_INDEX = 0
models = list(model_pricings.keys())
if 'model' not in st.session_state:
    st.session_state['model'] = models[DEFAULT_MODEL_INDEX]

PRICING = model_pricings[st.session_state['model']]

# przyjmuję w przybliżeniu kurs dolara 1$ = 4PLN
USD_TO_PLN = 4.0

# łączenie z OpenAI i instructorem
openai_client = OpenAI(api_key=st.session_state['openai_api_key'])
instructor_openai_client = instructor.from_openai(openai_client)

#
# konfiguracja 1. LLM (text -> JSON)
#

# automatyczne generowanie wyboru docelowej kolumny + KOSZTY + dekorator
class TargetColumn(BaseModel):
    target_column_name: str

@observe()
@st.cache_data
def retrieve_target_column_name(text, response_model):
    prompt = f"""
    Bądź jak Data Scientist!
    Na podstawie wyciągniętej listy zawierającej nazwy kolumn 
    wybierz kolumnę docelową.
    Kolumna docelowa jest najbardziej istotną kolumną w zbiorze danych.
    Kolumny: {text}.
    Przykład odpowiedzi:
    [
        {{"target_column_name": "species"}},
    ]
    """

    res, usage_info = instructor_openai_client.chat.completions.create_with_completion(
        model=st.session_state['model'],
        temperature=0,
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    usage = {}
    if usage_info.usage:
        usage = {
            'completion_tokens': usage_info.usage.completion_tokens,
            'prompt_tokens': usage_info.usage.prompt_tokens,
            'total_tokens': usage_info.usage.total_tokens,
        }

    result = {
        'content': res.model_dump(),
        'usage': usage,
    }

    if 'llm_messages' not in st.session_state:
        st.session_state['llm_messages'] = []
    st.session_state['llm_messages'].append(result)

    return result['content']

#
# konfiguracja 2. LLM (text -> JSON)
#

# automatyczne generowanie nazwy kolumn z krótkim opisem + KOSZTY + dekorator
class ColumnDescription(BaseModel):
    name: str
    description: str
 
@observe()
@st.cache_data
def retrieve_columns_info(text, response_model):
    prompt = f"""
    Na podstawie wyciągniętej listy zawierającej nazwy kolumn 
    przechwyć nazwę kolumny i dopasuj do niej 
    krótki opis po polsku 
    (zaczynając opis od małej litery i nie kończąc kropką) 
    Kolumny: {text}.
    Przykład odpowiedzi:
    [
        {{"name": "species", "description": "rodzaj irysa"}},
    ]
    """

    res, usage_info = instructor_openai_client.chat.completions.create_with_completion(
        model=st.session_state['model'],
        temperature=0,
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    usage = {}
    if usage_info.usage:
        usage = {
            'completion_tokens': usage_info.usage.completion_tokens,
            'prompt_tokens': usage_info.usage.prompt_tokens,
            'total_tokens': usage_info.usage.total_tokens,
        }

    result = {
        'content': res.model_dump(),
        'usage': usage,
    }

    if 'llm_messages' not in st.session_state:
        st.session_state['llm_messages'] = []
    st.session_state['llm_messages'].append(result)

    return result['content']

# iteracja po nazwach kolumn i wygenerowanie listy kolumn z ich krótkim opisem
@st.cache_data
def list_columns_info(column_names: List[str]) -> List[Dict]:
    descriptions = []  # lista do przechowywania wyników

    for column in column_names:
        # tworzenie prośby dla każdej kolumny zawartej w zbiorze danych
        response = retrieve_columns_info(
                                        text=column,
                                        response_model=ColumnDescription
                                        )
        # dodaj każdą odpowiedź do listy
        if response: 
            descriptions.append(response)

    return descriptions

#
# konfiguracja 3. LLM (image -> text)
#

# funkcja przygotowująca obraz do formatu binarnego rozumianego przez LLM
def prepare_image_for_llm(image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/png;base64,{image_data}'

# funkcja generująca opis wykresu przez LLM - z dekoratorem + KOSZTY + dekorator
@observe()
@st.cache_data
def describe_plot(image_path):
    res = openai_client.chat.completions.create(
        model=st.session_state['model'],
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Stwórz opis obrazka, skup się głównie na opisaniu dwóch najważniejszych cech. Opis wygeneruj tak jakby adresatem była osoba NIETECHNICZNA i dodaj na początku jedno zdanie ogólne na temat wykresu (czego dotyczą osie, co w skrócie przedstawia wykres)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_llm(image_path),
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    usage = {}
    if res.usage:
        usage = {
            'completion_tokens': res.usage.completion_tokens,
            'prompt_tokens': res.usage.prompt_tokens,
            'total_tokens': res.usage.total_tokens,
        }

    result = {
        'content': res.choices[0].message.content,
        'usage': usage,
    }

    if 'llm_messages' not in st.session_state:
        st.session_state['llm_messages'] = []
    st.session_state['llm_messages'].append(result)

    return result['content']

#
# konfiguracja 4. LLM (text -> text)
#

# funkcja generująca rekomendacje na podstawie opisu wykresu + KOSZTY + dekorator
@observe()
@st.cache_data
def generate_recommendations(plot_description):
    
    prompt = f"""
Wygeneruj rekomendacje dla klienta co może poprawić, aby 
zwiększyć efektywność dla docelowej kolumny w analizowanym zbiorze danych.
Potraktuj klienta jak osobę NIEtechniczną, czyli nie używaj terminologii związanej z modelami ML itp.
Np. mając zbiór danych dotyczących Titanic - kolumną docelową jest
liczba ocalałych survived. Wtedy wygeneruj rekomendacje na jakie inne
dane użytkownik ma wpłynąć (i w jaki sposób) aby więcej było
ocalałych osób. 
Rekomendacje wygeneruj na podstawie wygnerowanego opisu wykresu:
{plot_description}
"""
    
    res = openai_client.chat.completions.create(
        model=st.session_state['model'],
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    usage = {}
    if res.usage:
        usage = {
            'completion_tokens': res.usage.completion_tokens,
            'prompt_tokens': res.usage.prompt_tokens,
            'total_tokens': res.usage.total_tokens,
        }

    result = {
        'content': res.choices[0].message.content,
        'usage': usage,
    }

    if 'llm_messages' not in st.session_state:
        st.session_state['llm_messages'] = []
    st.session_state['llm_messages'].append(result)

    return result['content']


# inicjalizacja zmiennych w pamięci podręcznej
if 'regression' not in st.session_state:
    st.session_state['regression'] = False
if 'classification' not in st.session_state:
    st.session_state['classification'] = False
if 'upload_file' not in st.session_state:
    st.session_state['upload_file'] = False

# incjalizacja przycisków funkcyjnych
if 'go_to_target_column_button_clicked' not in st.session_state:
    st.session_state['go_to_target_column_button_clicked'] = False

if 'go_to_generate_ai_technology_button_clicked' not in st.session_state:
    st.session_state['go_to_generate_ai_technology_button_clicked'] = False

if 'generate_data_button_clicked' not in st.session_state:
    st.session_state['generate_data_button_clicked'] = False

# funkcja odpowiadająca za aktywację przycisków
def functional_button_click(button_name):
    st.session_state[button_name] = True

# pobieranie listy plików z przestrzeni Digital Ocean Spaces
@st.cache_resource
def get_dataset_names():
    response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_NAME)
    file_names = [item['Key'] for item in response.get('Contents', [])]

    # usunięcie prefiksu folderu i rozszerzenia
    dataset_names = [name[len(FOLDER_NAME):-4] for name in file_names if name.endswith('.csv')]
    return dataset_names

#
#MAIN
#

st.title('\U0001f575\uFE0F\u200D\u2642\uFE0F Detektyw najważniejszych zmiennych wpływających na wynik \U0001f575\U0001f3fd\u200D\u2640\uFE0F')

st.write('---')

st.markdown("""#### Witaj w aplikacji służącej do wykrywania najważniejszych zmiennych w zbiorze danych wpływających na wybraną kolumnę. <span style="color: green;">Aplikacja zasilana jest technologią AI, aby zmaksymalizować efektywność jej działania</span> \U0001f916<br><br> \u2B05\uFE0F Z lewej strony znajdują się informacje dotyczące stosowanego modelu AI wraz z kosztami użycia oraz możliwością przełączania się pomiędzy modelami. Dodatkowo dostępne jest tło muzyczne, aby umilić korzystanie z aplikacji \U0001f3b6""", unsafe_allow_html=True)

st.write('---')

st.markdown("#### Wgraj swoje dane w formacie CSV lub JSON albo skorzystaj z gotowego już zbioru danych")
with st.expander("Informacje dotyczące dodawania danych do analizy"):
    st.markdown(
        """
        * <span style="color: red;">Wczytywanie danych</span> - możesz wgrać dowolny plik CSV/JSON, który zostanie przetworzony na format możliwy do dalszej analizy

        * <span style="color: red;">Pobieranie danych</span> - możesz skorzystać z przykładowych gotowych już zestawów danych, aby przetestować funkcjonalność aplikacji
        """,
        unsafe_allow_html=True
    )

#
# v1 - wczytywanie pliku CSV i wyświetlanie przykładowych rekordów
#

#
# v8 - wczytywanie gotowych datasetów z Digital Ocean Spaces
#

# wycentrowanie zakładek za pomocą HTML i CSS
st.markdown(
    """
    <style>
    .stTabs [role="tablist"] {
        display: flex;
        width: 100%;
        justify-content: space-between;
    }
    .stTabs [role="tab"] {
        flex-grow: 1;
        text-align: center;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(['Wczytywanie danych', 'Pobieranie danych'])

# wczytywanie danych
with tab1:

    if not 'upload_file' in st.session_state:
        st.session_state['upload_file'] = None

    upload_file = st.file_uploader('Wybierz plik CSV lub JSON, który chcesz przeanalizować')

    # jeżeli wczytano plik
    if upload_file is not None:

        # wyciągam nazwę z pliku
        file_name = upload_file.name   

        #
        # v9 - dodanie możliwości wczytywania formatu JSON
        #

        if 'df' not in st.session_state:
            st.session_state['df'] = None
        
        # sprawdzam czy nazwa kończy się suffiksem: .csv
        if file_name.endswith('.csv') or file_name.endswith('.json'):

            # teraz aplikacja zapamięta wczytany plik - nawet po odświeżeniu
            st.session_state['upload_file'] = upload_file
            
            try:
                if file_name.endswith('.csv'):
                    df = pd.read_csv(st.session_state['upload_file'], sep=';')
                elif file_name.endswith('.json'):
                    df = pd.read_json(st.session_state['upload_file'])

                ## przypadek, gdy plik JSON jest pusty
                if df.empty:
                    st.error('Wczytany plik jest pusty. Wgraj prawidłowy plik CSV lub JSON.')
                    #st.session_state['df'] = df
                else:
                    st.session_state['df'] = df
                    st.markdown('#### Przykładowe 5 rekordów z wczytanego zbioru danych')
                    
                    # postawienie warunku - jeśli wczytany df ma mniej rekordów od 5
                    if len(st.session_state['df']) >= 5:
                        sample_df = st.session_state['df'].sample(5, random_state=123)
                    else:
                        sample_df = st.session_state['df']
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)
                    
                    columns_names = df.columns.tolist()
                    #with st.spinner('Trwa ładowanie danych...'):
                    columns_info = list_columns_info(columns_names)
                    
                    # wyjaśnienie nazw kolumn dla użytkownika
                    with st.expander("Informacje dotyczące zbioru danych"):

                        # sprawdzenie czy w przykładowym wczytanym zbiorze są NaN
                        if sample_df.isna().any().any():
                            text_for_user = '<span style="color: red;">None</span> - brakująca wartość (w danym miejscu we wczytanym zbiorze danych brakuje danych) <br>'
                        else:
                            text_for_user = ""
                        for item in columns_info:
                            text_for_user += f'<span style="color: red;">{item["name"]}</span> - {item["description"]} <br>'

                        st.markdown(text_for_user, unsafe_allow_html=True)
                    
                    if not st.session_state['go_to_target_column_button_clicked']:
                        st.button('Przejdź do wyboru kolumny', on_click = functional_button_click, 
                                args=('go_to_target_column_button_clicked', ), use_container_width=True)

                    if st.session_state['go_to_target_column_button_clicked']:
                    
                        st.write('---')

        #
        # v2 - wybieranie kolumny docelowej przez użytkownika
        #

        #
        # v6 - automatyczne wybieranie kolumny docelowej przez LLM
        #

                        st.markdown('#### Mając wczytany zbiór danych - wybierz teraz <span style="color: red;">metodę wyboru kolumny docelowej</span>, dla której zostanie przeprowadzona analiza:', unsafe_allow_html=True)            

                        # ustawienie przycisków wyboru metody    
                        if 'choice_method' not in st.session_state:
                            st.session_state['choice_method'] = 'select_option'
                        
                        if st.session_state['choice_method'] == 'select_option':

                            col1, col2 = st.columns(2)

                            with col1:

                                if st.button('Automatyczny wybór', use_container_width=True, help='AI zdecyduje za Ciebie'):
                                    st.session_state['choice_method'] = 'automatic'
                                    st.rerun()
                                
                            with col2:
                                
                                if st.button('Wybór użytkownika', use_container_width=True, help='Decyzja należy do Ciebie'):
                                    st.session_state['choice_method'] = 'manual'
                                    st.rerun()
                    
                        elif st.session_state['choice_method'] in ['automatic', 'manual']:
                        
                            if st.session_state['choice_method'] == 'automatic':
                                #with st.spinner('Trwa ładowanie danych...'):
                                automatic_choice = retrieve_target_column_name(columns_names, TargetColumn)
                                st.session_state['target_choice'] = automatic_choice['target_column_name']
                                
                                # wyświetlenie infomacji dla użytkownika - jaką kolumnę wybrał model AI
                                st.success(f'Automatycznie wybrana kolumna: {st.session_state["target_choice"]}')

                            elif st.session_state['choice_method'] == 'manual':
                                if 'target_choice' not in st.session_state:
                                    st.session_state['target_choice'] = columns_names[0]
                                target_choice = st.selectbox('Wybierz docelową kolumnę', columns_names)
                                
                                if target_choice != st.session_state['target_choice']:

                                    st.session_state['target_choice'] = target_choice
                                    
                                    # wyświetlenie informacji jaką kolumnę wybrał użytkownik - trwające przez 3 sek (jako potwierdzenie wyboru kolumny)
                                    success_message_container = st.empty()
                                    success_message_container.success(f'Wybrana przez Ciebie kolumna: {st.session_state["target_choice"]}')
                                    time.sleep(3)
                                    success_message_container.empty()

                            if st.button('Wróć do wyboru metody'):
                                del st.session_state['target_choice']
                                st.session_state['choice_method'] = 'select_option'
                                st.rerun()       

                            if st.session_state['target_choice']:

                                # czyszczenie danych - WARTOŚCI BRAKUJĄCE
                                sum_na_values = df[st.session_state['target_choice']].isna().sum()
                                missing_percent = df[st.session_state['target_choice']].isna().sum() / len(df[st.session_state['target_choice']]) * 100
                                
                                # przyjmuję, że WYPEŁNIAM wartości NaN jeśli ich ilość % NIE PRZEKRACZA 30% danych w kolumnie docelowej (typ numeryczny = mean, typ kategoryczny = mode)
                                # przyjmuję, że USUWAM wartości NaN jeśli ich ilość % PRZEKRACZA 30% danych w kolumnie docelowej

                                if sum_na_values != 0:
                                    if missing_percent <= 30:   
                                        if pd.api.types.is_numeric_dtype(df[st.session_state['target_choice']]):
                                            df[st.session_state['target_choice']].fillna(df[st.session_state['target_choice']].mean(), inplace=True)
                                        else:
                                            # wybieramy pierwszy element serii z funkcji mode()
                                            mode_value = df[st.session_state['target_choice']].mode()[0]
                                            df[st.session_state['target_choice']].fillna(mode_value, inplace=True)
                                    elif missing_percent > 30:
                                        df.dropna(subset=[st.session_state['target_choice']], inplace=True)

                                #
                                # v3 - sprawdzenie czy należy użyć modelu regresji czy klasyfikacji
                                #     

                                # klasyfikacja = typ NIEnumeryczny + numeryczny (gdzie liczba wartości unikalnych <= 10)
                                # regresja = typ numeryczny (gdzie liczba wartości unikalnych >10)
            
                                if not st.session_state['go_to_generate_ai_technology_button_clicked']:
                                    st.button('Przejdź do zastosowania technologii AI', on_click = functional_button_click, args=('go_to_generate_ai_technology_button_clicked', ), use_container_width=True)

                                if st.session_state['go_to_generate_ai_technology_button_clicked']:

                                    st.write('---')
                                    
                                    st.markdown(
                                        f'#### Wykorzystaj technologię AI do zbadania najważniejszych cech wpływających na kolumnę: <span style="color: red;">{st.session_state["target_choice"]}</span>',
                                        unsafe_allow_html=True)
                                    
                                    # sytuacja gdy typ kolumny jest numeryczny i liczba unikatowych wartości przekracza 10
                                    if (pd.api.types.is_numeric_dtype(df[st.session_state['target_choice']])) and (df[st.session_state['target_choice']].nunique() > 10):
                                        with st.expander('Informacje dotyczące stosowanej technologii AI'):
                                            st.markdown(
                                                'Do wygenerowania danych zastosowano <span style="color: red;">model regresji.</span> <br><br>Model regresyjny został użyty, aby określić, jakie czynniki wpływają na wartość, którą chcemy przewidzieć, oraz w jaki sposób to robią. <br><br>Możesz myśleć o tym jak o planowaniu budżetu domowego: chcemy zrozumieć, jak różne wydatki, takie jak rachunki, jedzenie czy rozrywka, wpływają na całkowite miesięczne wydatki. Model regresyjny działa jak narzędzie, które pomaga zobaczyć, jak zmiana w jednym z tych wydatków może wpłynąć na cały budżet. Dzięki temu możemy lepiej zarządzać finansami, wiedząc, które aspekty mają największy wpływ na nasze koszty.',
                                                unsafe_allow_html=True)
                                    #st.metric('Mamy do czynienia z problemem:', 'Regresji')
                                        st.session_state['regression'] = True
                                        st.session_state['classification'] = False
                                    
                                    # każda pozostała sytuacja wygeneruje model klasyfikacji
                                    else:
                                        with st.expander('Informacje dotyczące stosowanej technologii'):
                                            st.markdown(
                                                'Do wygenerowania danych zastosowano <span style="color: red;">model klasyfikacji.</span> <br><br>Model klasyfikacji został użyty, aby pomóc zrozumieć, które cechy lub właściwości mają największy wpływ na to, co próbujemy przewidzieć lub klasyfikować. <br><br>Wyobraź sobie, że chcesz dowiedzieć się, które składniki przepisu na ciasto najbardziej wpływają na to, czy ciasto wyjdzie smaczne. Model klasyfikacji działa jak doświadczony kucharz, który podpowiada, które składniki są najważniejsze dla uzyskania pysznego rezultatu. Dzięki temu możemy skupić się na tych kluczowych składnikach, które mają największe znaczenie dla ostatecznego wyniku.',
                                                unsafe_allow_html=True)
                                    #st.metric('Mamy do czynienia z problemem:', 'Klasyfikacji')
                                        st.session_state['classification'] = True
                                        st.session_state['regression'] = False 
                                    
                                    if not st.session_state['generate_data_button_clicked']:
                                        st.button('Generuj dane', on_click= functional_button_click, args=('generate_data_button_clicked', ), use_container_width=True)
                                    
                                
                                    # lista sucharów dla umilenia czasu oczekiwania na model :)
                                    jokes = [
                                    "Dlaczego programista nie lubi chodzić do lasu? Bo boi się zgubić w gąszczu kodu.",
                                    "Jak programiści liczą do trzech? Zero, jeden, dwa, trzy.",
                                    "Jakie jest ulubione drzewo programisty? Drzewo binarne.",
                                    "Dlaczego programista nie może się zdecydować? Bo zawsze ma if i else.",
                                    "Czemu komputer się przeziębił? Bo miał za dużo otwartych okien.",
                                    "Jakie jest ulubione jedzenie programistów? Spaghetti... kod!",
                                    "Dlaczego programista nie może znaleźć kluczy? Bo szuka w niewłaściwej pętli.",
                                    "Dlaczego programiści nie lubią gotować? Bo boją się wycieku pamięci.",
                                    "Dlaczego komputerowi nie wolno ufać? Bo może wyciec z niego informacja!",
                                    "Dlaczego programiści nie lubią natury? Bo jest zbyt wiele bugów.",
                                    ]
                                
                                    if st.session_state['generate_data_button_clicked']:

                                        #
                                        # v7 - rekomendacje dla klienta na podstawie uzyskanego opisu (model LLM: text->text)
                                        #

                                        #
                                        # auto ML - klasyfikacja
                                        #

                                        if st.session_state['classification']:
                                            class_value_counts = df[st.session_state["target_choice"]].value_counts()

                                            # sprawdzenie, czy istnieje dowolna klasa z tylko jednym przedstawicielem -> JEŚLI TAK, nie możemy zastosować modelu klasyfikacji
                                            has_single_member_class = any(class_value_counts == 1)

                                            if has_single_member_class:
                                                st.error('Dane w tej kolumnie nie nadają się do klasyfikacji. Wybierz inną kolumnę!')
                                                with st.expander('Informacje dotyczące wybranej kolumny'):
                                                    st.markdown(
                                                    '<span style="color: red;">W tej kolumnie znajduje się minimum jedna dana, która posiada tylko jednego przedstawiciela. Co to oznacza w praktyce?</span> <br><br>Wyobraź sobie, że uczysz się rozpoznawać różne rodzaje owoców patrząc na przykłady i porównując je. Jeśli masz do nauki tylko jedno jabłko i właśnie to jedno musisz zapamiętać jako przykład jabłka, może być trudno rozpoznać inne jabłka, które wyglądają choćby odrobinę inaczej.<br><br>Podobnie jest z modelami klasyfikacji w analizie danych. Modele te "uczą się", jakie cechy odróżniają jedną klasę od drugiej, analizując wiele przykładów z każdej klasy. Jeśli masz tylko jeden przykład z danej klasy, model nie ma wystarczająco dużo informacji, aby dobrze nauczyć się, jak rozróżniać tę klasę od innych. Bez większej liczby przykładów, istnieje ryzyko, że model błędnie sklasyfikuje rzeczy, które nie są do końca takie same, albo że zignoruje tę klasę całkowicie w przyszłych przewidywaniach. To tak, jakby ucząc się tylko z jednego jabłka, można było przeoczyć fakt, że jabłka mogą być również zielone czy większe niż to jedno, które widzisz.',
                                                    unsafe_allow_html=True)
                                                st.stop()

                                            else:
                                                
                                                # konfiguracja
                                                cls_setup(data=df, target=st.session_state['target_choice'], session_id=123)
                                                
                                                joke_placeholder = st.empty()

                                                with st.spinner("Trwa generowanie danych przez AI..."):
                                                    
                                                    # generowanie losowego kawału podczas oczekiwania na wynik
                                                    joke_placeholder.text(random.choice(jokes))
                                                    
                                                    # znalezienie najlepszego modelu - z wykluczeniem NIEobsługujących wykresu 'feature' !!
                                                    best_classify_model = cls_compare_models(exclude=['knn', 'svm', 'gpc', 'nb'])
                                                
                                                joke_placeholder.empty()

                                            # sprawdzenie czy znaleziony BEST MODEL posiada możliwość generowania wykresu funkcji
                                            if hasattr(best_classify_model, 'coef_') or hasattr(best_classify_model, 'feature_importances_'):
                                                
                                                cls_plot_model(best_classify_model, plot='feature', save=True, verbose= False)
                                                with open('Feature Importance.png', 'rb') as f:
                                                    st.session_state['feature_importance'] = f.read()

                                                st.image(st.session_state['feature_importance'])
                                                
                                                st.stop()

                                                # # Zakładamy, że model jest już utworzony i dostępny jako `model`
                                                # def generate_feature_importance():
                                                #     cls_plot_model(best_classify_model, plot='feature', save=True, verbose=False)

                                                # # Generuj wykres i zapisuj go jako PNG
                                                # generate_feature_importance()

                                                # # Otwórz plik PNG i wczytaj jego zawartość
                                                # with open("Feature Importance.png", "rb") as file:
                                                #     btn = st.download_button(
                                                #         label="Pobierz Feature Importance",
                                                #         data=file,
                                                #         file_name="Feature_Importance.png",
                                                #         mime="image/png"
                                                #         #test
                                                #     )
                                                # st.image('Feature Importance.png')
                                                
                                                # # wygenerowanie wykresu istotności cech
                                                # cls_plot_model(best_classify_model, plot='feature', save=True)

                                                # # zapisanie wykresu do lokalnej zmiennej
                                                # local_plot_path = 'Feature Importance.png'
                                                
                                                # # wykorzystanie buffora do wyświetlenia wykresu
                                                # with open(local_plot_path, 'rb') as f:
                                                #     buf = io.BytesIO(f.read())
                                                # #st.image(buf)
                                                # # buf = io.BytesIO()
                                                # # plt.savefig(buf, format='png')
                                                # # buf.seek(0)
                                                # #if not cls_plot.empty:
                                                # client.put_object(Bucket=BUCKET_NAME, Key=f'{CLS_PLOT_FOLDER_NAME}{PLOT_NAME}', Body=buf)
                                                
                                                # response = client.get_object(Bucket=BUCKET_NAME, Key=f'{CLS_PLOT_FOLDER_NAME}{PLOT_NAME}')
                                                # plot_data = response['Body'].read()

                                                # st.image(io.BytesIO(plot_data), use_column_width=True)

                                    

                          

                                                # else:
                                                #     st.error('Wykres nie został odnaleziony.')
                                            else:
                                                st.error(
                                                'Wygenerowanie wykresu istotności cech NIE jest możliwe dla tej kolumny. Zmień kolumnę docelową.')
                                                
                                        
                                        #
                                        # autoML - model regresji
                                        #

                                        elif st.session_state['regression']:
                                            
                                            # konfiguracja
                                            reg_setup(data=df, target=st.session_state['target_choice'], session_id=123)
                                        
                                            joke_placeholder = st.empty()

                                            with st.spinner("Trwa generowanie danych przez AI..."):
                                                
                                                # generowanie losowego kawału podczas oczekiwania na wynik
                                                joke_placeholder.text(random.choice(jokes))
                                                
                                                # znalezienie najlepszego modelu - z wykluczeniem NIEobsługujących wykresu 'feature' !!
                                                best_regress_model = reg_compare_models(exclude=['knn', 'svm', 'gpc', 'nb'])
                                                
                                            joke_placeholder.empty()

                                            # sprawdzenie czy znaleziony BEST MODEL posiada możliwość generowania wykresu funkcji
                                            if hasattr(best_regress_model, 'coef_') or hasattr(best_regress_model, 'feature_importances_'):
                                                
                                                # korzystam z tymczasowej zmiennej zeby obluzyc dzialanie z Digital Ocean App
                                                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                                    plot_file_path = tmp_file.name

                                                # Wygeneruj i zapisz wykres
                                                reg_plot_model(best_regress_model, plot='feature', save=True)
                                                os.rename('Feature Importance.png', plot_file_path)

                                                # Wyświetl wykres w Streamlit
                                                st.image(plot_file_path, use_column_width=True)

                                                # Opis wykresu
                                                st.markdown('#### Opis wykresu:')
                                                reg_description = describe_plot(plot_file_path)
                                                st.write(reg_description)

                                                # Rekomendacje
                                                st.markdown('#### <span style="color: green;">Rekomendacje:</span>', unsafe_allow_html=True)
                                                st.write(generate_recommendations(reg_description))

                                                # Usuń plik tymczasowy
                                                os.remove(plot_file_path)
                                                
                                            else:
                                                st.error(
                                                    'Wygenerowanie wykresu istotności cech NIE jest możliwe dla tej kolumny. Zmień kolumnę docelową.')

            # przypadek, gdy plik CSV jest pusty
            except pd.errors.EmptyDataError:
                st.error('Wczytany plik jest pusty. Wgraj prawidłowy plik CSV lub JSON.')
            # sytuacja, np. gdy ktoś wrzuca pusty plik JSON -> prawidłowo powinien zawierać {} aby być traktowany jako pusty
            except ValueError as e:
                st.error(f'Błąd odczytu pliku: {e}. Upewnij się, że plik jest w poprawnym formacie.')
            # przypadek, gdy nie wgrano żadnego pliku
        
        # przypadek, gdy wgrany plik jest innego typu niż CSV
        else:
            st.error('Nieprawidłowy format pliku - wgraj plik CSV lub JSON')

    else:
        st.info('Nie wczytano żadnego pliku')

# pobieranie danych
with tab2:

    datasets = get_dataset_names()
    selected_dataset = st.selectbox("Wybierz dataset", ['Wybierz z listy'] + datasets)

    if selected_dataset != 'Wybierz z listy':

        try:
            data_url = f's3://{BUCKET_NAME}/{FOLDER_NAME}{selected_dataset}.csv'
            ready_df = pd.read_csv(data_url, sep=',')
            st.session_state['upload_file'] = ready_df
            #st.write(f"Wybrano dataset: {selected_dataset}")
            st.markdown('#### Przykładowe 5 rekordów z wczytanego zbioru danych')
            sample_ready_df = ready_df.sample(5, random_state=123)
            st.dataframe(sample_ready_df, use_container_width=True, hide_index=True)
            
            columns_names = ready_df.columns.tolist()
            #with st.spinner('Trwa ładowanie danych...'):
            columns_info = list_columns_info(columns_names)
            
            # wyjaśnienie nazw kolumn dla użytkownika
            with st.expander("Informacje dotyczące zbioru danych"):

                # sprawdzenie czy w przykładowym wczytanym zbiorze są NaN
                if sample_ready_df.isna().any().any():
                    text_for_user = '<span style="color: red;">None</span> - brakująca wartość (w danym miejscu we wczytanym zbiorze danych brakuje danych) <br>'
                else:
                    text_for_user = ""
                for item in columns_info:
                    text_for_user += f'<span style="color: red;">{item["name"]}</span> - {item["description"]} <br>'

                st.markdown(text_for_user, unsafe_allow_html=True)
            
            if not st.session_state['go_to_target_column_button_clicked']:
                st.button('Przejdź do wyboru kolumny ', on_click = functional_button_click, 
                        args=('go_to_target_column_button_clicked', ), use_container_width=True)

            if st.session_state['go_to_target_column_button_clicked']:
            
                st.write('---')

#
# v2 - wybieranie kolumny docelowej przez użytkownika
#

#
# v6 - automatyczne wybieranie kolumny docelowej przez LLM
#

                st.markdown('#### Mając wczytany zbiór danych - wybierz teraz <span style="color: red;">metodę wyboru kolumny docelowej</span>, dla której zostanie przeprowadzona analiza:', unsafe_allow_html=True)            

                # ustawienie przycisków wyboru metody     
                if 'choice_method' not in st.session_state:
                    st.session_state['choice_method'] = 'select_option'
                
                if st.session_state['choice_method'] == 'select_option':

                    col1, col2 = st.columns(2)

                    with col1:

                        if st.button('Automatyczny wybór ', use_container_width=True, help='AI zdecyduje za Ciebie'):
                            st.session_state['choice_method'] = 'automatic'
                            st.rerun()
                        
                    with col2:
                        
                        if st.button('Wybór użytkownika ', use_container_width=True, help='Decyzja należy do Ciebie'):
                            st.session_state['choice_method'] = 'manual'
                            st.rerun()
            
                elif st.session_state['choice_method'] in ['automatic', 'manual']:
                
                    if st.session_state['choice_method'] == 'automatic':
                        #with st.spinner('Trwa ładowanie danych...'):
                        automatic_choice = retrieve_target_column_name(columns_names, TargetColumn)
                        st.session_state['target_choice'] = automatic_choice['target_column_name']
                        
                        # wyświetlenie infomacji dla użytkownika - jaką kolumnę wybrał model AI
                        st.success(f'Automatycznie wybrana kolumna: {st.session_state["target_choice"]}')

                    elif st.session_state['choice_method'] == 'manual':
                        if 'target_choice' not in st.session_state:
                            st.session_state['target_choice'] = columns_names[0]
                        target_choice = st.selectbox('Wybierz docelową kolumnę', columns_names)
                        
                        if target_choice != st.session_state['target_choice']:

                            st.session_state['target_choice'] = target_choice
                            
                            # wyświetlenie informacji jaką kolumnę wybrał użytkownik - trwające przez 3 sek (jako potwierdzenie wyboru kolumny)
                            success_message_container = st.empty()
                            success_message_container.success(f'Wybrana przez Ciebie kolumna: {st.session_state["target_choice"]}')
                            time.sleep(3)
                            success_message_container.empty()

                    if st.button('Wróć do wyboru metody '):
                        del st.session_state['target_choice']
                        st.session_state['choice_method'] = 'select_option'
                        st.rerun()       

                    if st.session_state['target_choice']:

                        # czyszczenie danych - WARTOŚCI BRAKUJĄCE
                        sum_na_values = ready_df[st.session_state['target_choice']].isna().sum()
                        missing_percent = ready_df[st.session_state['target_choice']].isna().sum() / len(ready_df[st.session_state['target_choice']]) * 100
                        
                        # przyjmuję, że WYPEŁNIAM wartości NaN jeśli ich ilość % NIE PRZEKRACZA 30% danych w kolumnie docelowej (typ numeryczny = mean, typ kategoryczny = mode)
                        # przyjmuję, że USUWAM wartości NaN jeśli ich ilość % PRZEKRACZA 30% danych w kolumnie docelowej

                        if sum_na_values != 0:
                            if missing_percent <= 30:   
                                if pd.api.types.is_numeric_dtype(ready_df[st.session_state['target_choice']]):
                                    ready_df[st.session_state['target_choice']].fillna(ready_df[st.session_state['target_choice']].mean(), inplace=True)
                                else:
                                    # wybieramy pierwszy element serii z funkcji mode()
                                    mode_value = ready_df[st.session_state['target_choice']].mode()[0]
                                    ready_df[st.session_state['target_choice']].fillna(mode_value, inplace=True)
                            elif missing_percent > 30:
                                ready_df.dropna(subset=[st.session_state['target_choice']], inplace=True)

                        #
                        # v3 - sprawdzenie czy należy użyć modelu regresji czy klasyfikacji
                        #     

                        # klasyfikacja = typ NIEnumeryczny + numeryczny (gdzie liczba wartości unikalnych <= 10)
                        # regresja = typ numeryczny (gdzie liczba wartości unikalnych >10)
    
                        if not st.session_state['go_to_generate_ai_technology_button_clicked']:
                            st.button('Przejdź do zastosowania technologii AI ', on_click = functional_button_click, args=('go_to_generate_ai_technology_button_clicked', ), use_container_width=True)

                        if st.session_state['go_to_generate_ai_technology_button_clicked']:

                            st.write('---')
                            
                            st.markdown(
                                f'#### Wykorzystaj technologię AI do zbadania najważniejszych cech wpływających na kolumnę: <span style="color: red;">{st.session_state["target_choice"]}</span>',
                                unsafe_allow_html=True)
                            
                            # sytuacja gdy typ kolumny jest numeryczny i liczba unikatowych wartości przekracza 10 
                            if (pd.api.types.is_numeric_dtype(ready_df[st.session_state['target_choice']])) and (ready_df[st.session_state['target_choice']].nunique() > 10):
                                with st.expander('Informacje dotyczące stosowanej technologii AI'):
                                    st.markdown(
                                        'Do wygenerowania danych zastosowano <span style="color: red;">model regresji.</span> <br><br>Model regresyjny został użyty, aby określić, jakie czynniki wpływają na wartość, którą chcemy przewidzieć, oraz w jaki sposób to robią. <br><br>Możesz myśleć o tym jak o planowaniu budżetu domowego: chcemy zrozumieć, jak różne wydatki, takie jak rachunki, jedzenie czy rozrywka, wpływają na całkowite miesięczne wydatki. Model regresyjny działa jak narzędzie, które pomaga zobaczyć, jak zmiana w jednym z tych wydatków może wpłynąć na cały budżet. Dzięki temu możemy lepiej zarządzać finansami, wiedząc, które aspekty mają największy wpływ na nasze koszty.',
                                        unsafe_allow_html=True)
                            #st.metric('Mamy do czynienia z problemem:', 'Regresji')
                                st.session_state['regression'] = True
                                st.session_state['classification'] = False
                            
                            # każda pozostała sytuacja wygeneruje model klasyfikacji      
                            else:
                                with st.expander('Informacje dotyczące stosowanej technologii'):
                                    st.markdown(
                                        'Do wygenerowania danych zastosowano <span style="color: red;">model klasyfikacji.</span> <br><br>Model klasyfikacji został użyty, aby pomóc zrozumieć, które cechy lub właściwości mają największy wpływ na to, co próbujemy przewidzieć lub klasyfikować. <br><br>Wyobraź sobie, że chcesz dowiedzieć się, które składniki przepisu na ciasto najbardziej wpływają na to, czy ciasto wyjdzie smaczne. Model klasyfikacji działa jak doświadczony kucharz, który podpowiada, które składniki są najważniejsze dla uzyskania pysznego rezultatu. Dzięki temu możemy skupić się na tych kluczowych składnikach, które mają największe znaczenie dla ostatecznego wyniku.',
                                        unsafe_allow_html=True)
                            #st.metric('Mamy do czynienia z problemem:', 'Klasyfikacji')
                                st.session_state['classification'] = True
                                st.session_state['regression'] = False 
                            
                            if not st.session_state['generate_data_button_clicked']:
                                st.button('Generuj dane ', on_click= functional_button_click, args=('generate_data_button_clicked', ), use_container_width=True)
                            
                        
                            # lista sucharów dla umilenia czasu oczekiwania na model :)
                            jokes = [
                            "Dlaczego programista nie lubi chodzić do lasu? Bo boi się zgubić w gąszczu kodu.",
                            "Jak programiści liczą do trzech? Zero, jeden, dwa, trzy.",
                            "Jakie jest ulubione drzewo programisty? Drzewo binarne.",
                            "Dlaczego programista nie może się zdecydować? Bo zawsze ma if i else.",
                            "Czemu komputer się przeziębił? Bo miał za dużo otwartych okien.",
                            "Jakie jest ulubione jedzenie programistów? Spaghetti... kod!",
                            "Dlaczego programista nie może znaleźć kluczy? Bo szuka w niewłaściwej pętli.",
                            "Dlaczego programiści nie lubią gotować? Bo boją się wycieku pamięci.",
                            "Dlaczego komputerowi nie wolno ufać? Bo może wyciec z niego informacja!",
                            "Dlaczego programiści nie lubią natury? Bo jest zbyt wiele bugów.",
                            ]
                        
                            if st.session_state['generate_data_button_clicked']:

                                #
                                # v7 - rekomendacje dla klienta na podstawie uzyskanego opisu (model LLM: text->text)
                                #

                                #
                                # auto ML - klasyfikacja
                                #

                                if st.session_state['classification']:
                                    class_value_counts = ready_df[st.session_state["target_choice"]].value_counts()

                                    # sprawdzenie, czy istnieje dowolna klasa z tylko jednym przedstawicielem -> JEŚLI TAK, nie możemy zastosować modelu klasyfikacji
                                    has_single_member_class = any(class_value_counts == 1)

                                    if has_single_member_class:
                                        st.error('Dane w tej kolumnie nie nadają się do klasyfikacji. Wybierz inną kolumnę!')
                                        with st.expander('Informacje dotyczące wybranej kolumny'):
                                            st.markdown(
                                            '<span style="color: red;">W tej kolumnie znajduje się minimum jedna dana, która posiada tylko jednego przedstawiciela. Co to oznacza w praktyce?</span> <br><br>Wyobraź sobie, że uczysz się rozpoznawać różne rodzaje owoców patrząc na przykłady i porównując je. Jeśli masz do nauki tylko jedno jabłko i właśnie to jedno musisz zapamiętać jako przykład jabłka, może być trudno rozpoznać inne jabłka, które wyglądają choćby odrobinę inaczej.<br><br>Podobnie jest z modelami klasyfikacji w analizie danych. Modele te "uczą się", jakie cechy odróżniają jedną klasę od drugiej, analizując wiele przykładów z każdej klasy. Jeśli masz tylko jeden przykład z danej klasy, model nie ma wystarczająco dużo informacji, aby dobrze nauczyć się, jak rozróżniać tę klasę od innych. Bez większej liczby przykładów, istnieje ryzyko, że model błędnie sklasyfikuje rzeczy, które nie są do końca takie same, albo że zignoruje tę klasę całkowicie w przyszłych przewidywaniach. To tak, jakby ucząc się tylko z jednego jabłka, można było przeoczyć fakt, że jabłka mogą być również zielone czy większe niż to jedno, które widzisz.',
                                            unsafe_allow_html=True)
                                        st.stop()

                                    else:
                                        
                                        # konfiguracja
                                        cls_setup(data=ready_df, target=st.session_state['target_choice'], session_id=123)
                                        
                                        joke_placeholder = st.empty()

                                        with st.spinner("Trwa generowanie danych przez AI..."):
                                            
                                            # generowanie losowego kawału podczas oczekiwania na wynik
                                            joke_placeholder.text(random.choice(jokes))
                                            
                                            # znalezienie najlepszego modelu - z wykluczeniem NIEobsługujących wykresu 'feature' !!
                                            best_classify_model = cls_compare_models(exclude=['knn', 'svm', 'gpc', 'nb'])
                                        
                                        joke_placeholder.empty()

                                    # sprawdzenie czy znaleziony BEST MODEL posiada możliwość generowania wykresu funkcji
                                    if hasattr(best_classify_model, 'coef_') or hasattr(best_classify_model, 'feature_importances_'):

                                    
                                        
                                        cls_plot_model(best_classify_model, plot='feature', save=True, verbose= False)
                                        with open('Feature Importance.png', 'rb') as f:
                                            st.session_state['feature_importance'] = f.read()

                                        st.image(st.session_state['feature_importance'])
                                        
                                        st.stop()
                                        
                                        # # korzystam z tymczasowej zmiennej zeby obluzyc dzialanie z Digital Ocean App
                                        # with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                        #             plot_file_path = tmp_file.name

                                        # # Wygeneruj i zapisz wykres
                                        # cls_plot_model(best_classify_model, plot='feature', save=True)
                                        # os.rename('Feature Importance.png', plot_file_path)

                                        # # Wyświetl wykres w Streamlit
                                        # st.image(plot_file_path, use_column_width=True)

                                        # # Opis wykresu
                                        # st.markdown('#### Opis wykresu:')
                                        # cls_description = describe_plot(plot_file_path)
                                        # st.write(cls_description)

                                        # # Rekomendacje
                                        # st.markdown('#### <span style="color: green;">Rekomendacje:</span>', unsafe_allow_html=True)
                                        # st.write(generate_recommendations(cls_description))

                                        # # Usuń plik tymczasowy
                                        # os.remove(plot_file_path)

                                    else:
                                        st.error(
                                        'Wygenerowanie wykresu istotności cech NIE jest możliwe dla tej kolumny. Zmień kolumnę docelową.')

                                #
                                # autoML - model regresji
                                #

                                elif st.session_state['regression']:
                                    
                                    # konfiguracja
                                    reg_setup(data=ready_df, target=st.session_state['target_choice'], session_id=123)
                                
                                    joke_placeholder = st.empty()

                                    with st.spinner("Trwa generowanie danych przez AI..."):
                                        
                                        # generowanie losowego kawału podczas oczekiwania na wynik
                                        joke_placeholder.text(random.choice(jokes))
                                        
                                        # znalezienie najlepszego modelu - z wykluczeniem NIEobsługujących wykresu 'feature' !!
                                        best_regress_model = reg_compare_models(exclude=['knn', 'svm', 'gpc', 'nb'])
                                        
                                    joke_placeholder.empty()

                                    # sprawdzenie czy znaleziony BEST MODEL posiada możliwość generowania wykresu funkcji
                                    if hasattr(best_regress_model, 'coef_') or hasattr(best_regress_model, 'feature_importances_'):
                                        
                                        # korzystam z tymczasowej zmiennej zeby obluzyc dzialanie z Digital Ocean App
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                                    plot_file_path = tmp_file.name

                                        # Wygeneruj i zapisz wykres
                                        reg_plot_model(best_regress_model, plot='feature', save=True)
                                        os.rename('Feature Importance.png', plot_file_path)

                                        # Wyświetl wykres w Streamlit
                                        st.image(plot_file_path, use_column_width=True)

                                        # Opis wykresu
                                        st.markdown('#### Opis wykresu:')
                                        reg_description = describe_plot(plot_file_path)
                                        st.write(reg_description)

                                        # Rekomendacje
                                        st.markdown('#### <span style="color: green;">Rekomendacje:</span>', unsafe_allow_html=True)
                                        st.write(generate_recommendations(reg_description))

                                        # Usuń plik tymczasowy
                                        os.remove(plot_file_path)

                                    else:
                                        st.error(
                                            'Wygenerowanie wykresu istotności cech NIE jest możliwe dla tej kolumny. Zmień kolumnę docelową.')

        # przypadek, gdy plik CSV jest pusty
        except pd.errors.EmptyDataError:
            st.error('Wczytany plik wydaje się być pusty. Spróbuj wgrać inny plik!')

#
# PASEK BOCZNY
#

with st.sidebar:
    st.write(f'Aktualny model: {st.session_state["model"]}')

    # dodanie opcji wyboru modelu LLM
    selected_model = st.selectbox('Wybrany model', models, index=DEFAULT_MODEL_INDEX)
    st.session_state['model'] = selected_model

    # wyświetlenie kosztów użycia
    total_cost = 0.0
    for llm_message in st.session_state.get('llm_messages', []):
        if 'usage' in llm_message:
            total_cost += llm_message['usage'].get('prompt_tokens', 0) * PRICING['input_tokens']
            total_cost += llm_message['usage'].get('completion_tokens', 0) * PRICING['output_tokens']

    c0, c1 = st.columns(2)
    with c0:
        st.metric('Koszt użycia AI (USD)', f'${total_cost:.4f}')

    with c1:
        st.metric('Koszt użycia AI (PLN)', f'{total_cost * USD_TO_PLN:.4f}')
    with st.expander('Przyjęty kurs dolara'):
        st.write('Przyjęto w przybliżeniu, że 1$ = 4PLN')

    # wyświetlenie dostępnej muzyki
    st.write("---")
    st.markdown('##### Dostępne tło muzyczne \U0001f3b6')

    selected_song = st.selectbox('Wybierz utwór', list(songs.keys()))
    
    if selected_song:
        file_name = songs[selected_song]
        mp3_file = get_mp3_file(BUCKET_NAME, MUSIC_FOLDER_NAME, file_name)
        # odtwarzacz muzyczny
        st.audio(BytesIO(mp3_file), format='audio/mp3')


    


                            