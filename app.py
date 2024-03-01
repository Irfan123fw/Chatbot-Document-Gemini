from navigation import make_sidebar
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import matplotlib.pyplot as plt
import numpy as np
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import time
import hashlib
import os
import shutil
import gzip
from time import sleep
# Directory where the database file is stored
db_dir = 'database'
db_filename = 'user_runs.json'
db_path = os.path.join(db_dir, db_filename)
# Initialize or load the database
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Check if the database file already exists
if not os.path.isfile(db_path):
    user_runs = {}
else:
    with open(db_path, 'r') as db_file:
        user_runs = json.load(db_file)

# Function to check the maximum number of runs for a user
def check_max_runs(username):
    if username in user_runs:
        user_info = user_runs[username]
        if user_info['runs'] >= 3:
            return False, "Maximum number of runs reached."
    return True, None

# Function to update user runs
def update_user_runs(username, method):
    if username in user_runs:
        user_info = user_runs[username]
        user_info['runs'] += 1
        if method not in user_info['methods']:
            user_info['methods'].append(method)
    else:
        user_runs[username] = {'runs': 1, 'methods': [method]}
    with open(db_path, 'w') as db_file:
        json.dump(user_runs, db_file)
    return f"Run {user_runs[username]['runs']}"


# Function to save performance metrics
def save_performance_metrics(username,unique_id, method, run_number, precision, recall, f1):
    with open('score.txt', 'a') as score_file:
        score_file.write(f"{username},{unique_id},{method},{run_number},{precision},{recall},{f1}\n")
# Directory where you want to save the uploaded files
save_dir = 'uploaded_files'
make_sidebar()
if st.session_state.get('logged_in', False):
    st.info('Here is an example of the data: https://github.com/Irfan123fw/SentimenClassification/blob/main/yelp_labelled.txt', icon="ℹ️")

    # Access the username from the session state
    username = st.session_state.username
    # Check if the user can run the model
    can_run, message = check_max_runs(username)
    if not can_run:
        st.error(message)
        st.stop()

    method  = st.text_input("Metode", placeholder="Masukkan metode apa yag anda gunakan.")
    if method == "":
        st.warning("Please enter a method.")
        st.stop()

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)

    for uploaded_file in uploaded_files:
        unique_id = hashlib.sha256((uploaded_file.name + str(time.time())).encode()).hexdigest()
        print(f'Unique ID: {unique_id}')
        file_dir = os.path.join(save_dir, unique_id)
        os.makedirs(file_dir, exist_ok=True)

        # Save the uploaded file to this directory
        with gzip.open(os.path.join(file_dir, uploaded_file.name + '.gz'), 'wb') as f:
            f.write(uploaded_file.getvalue())

            
        if uploaded_file.name.endswith('.txt'):
            # Read the TXT file into a DataFrame
            df = pd.read_csv(uploaded_file, names=['sentence', 'label'], sep='\t')
        else:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)        
        tbl = pd.DataFrame(df)
        # st.table(tbl)
        df['sentence'] = df['sentence'].str.lower()

        df.head()
        stop_word = set(stopwords.words('english'))

        df['sentence'] = df['sentence'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_word)]))

        df.head()
        sentence = df['sentence'].values
        label = df['label'].values
        sentence_train, sentence_test, label_train, label_test = train_test_split(sentence, label, test_size=0.2, shuffle=False)

        print('Training dataset:\n', sentence_train.shape, label_train.shape)
        print('\nTest dataset:\n', sentence_test.shape, label_test.shape)
        filt = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ' # Untuk menghilangkan symbols

        tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>", filters=filt)

        tokenizer.fit_on_texts(sentence_train)
        word_index = tokenizer.word_index
        print(len(word_index))

        with open('word_index.json', 'w') as fp:
            json.dump(word_index, fp)

        train_sekuens = tokenizer.texts_to_sequences(sentence_train)
        test_sekuens = tokenizer.texts_to_sequences(sentence_test)


        train_padded = pad_sequences(train_sekuens, 
                                    maxlen=20,
                                    padding='post',
                                    truncating='post')
        test_padded = pad_sequences(test_sekuens,
                                    maxlen=20,
                                    padding='post',
                                    truncating='post')
        model = tf.keras.Sequential([
            Embedding(2000, 20, input_length=20),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.summary()
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        num_epochs = 30
        history = model.fit(train_padded, label_train, 
                            epochs=num_epochs, 
                            validation_data=(test_padded, label_test),
                            verbose=1)
        # Make predictions on the test set
        predictions = model.predict(test_padded)
        predictions = [1 if p > 0.5 else 0 for p in predictions]

        # Calculate Precision, Recall and F1 Score
        prc = precision_score(label_test, predictions)
        rec = recall_score(label_test, predictions)
        f1s = f1_score(label_test, predictions)
        precision = f'Precision: {prc}'
        recall = f'Recall: {rec}'
        f1 = f'F1 Score: {f1s}'
        message = update_user_runs(username, method)
        run_number = message
        save_performance_metrics(username,unique_id, method, run_number, prc, rec, f1s)
        
        # Display the performance metrics
        st.write(precision)
        st.write(recall)
        st.write(f1)
        sleep(3)
        st.switch_page("pages/score.py")
else:
    st.error("You are not logged in.")
    st.switch_page("./app.py")
