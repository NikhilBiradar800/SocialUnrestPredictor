# Importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading dataset
dataset = pd.read_csv('Emotion_classify_Data.csv')
print(dataset.info())

# Converting the text column to lowercase
dataset['Comment'] = dataset['Comment'].str.lower()

# Text cleaning
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in word_tokenize(text) if word not in set(stopwords.words('english')))
    return text

# Applying text cleaning to 'Comment' column
dataset['Comment'] = dataset['Comment'].apply(clean_text)
# Split the dataset into training, validation, and test sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Labelling Encoding step
label_encoder = LabelEncoder()
label_encoder.fit(dataset['Emotion'])  # Fit on the entire label space
train_data['Emotion'] = label_encoder.transform(train_data['Emotion'])
val_data['Emotion'] = label_encoder.transform(val_data['Emotion'])
test_data['Emotion'] = label_encoder.transform(test_data['Emotion'])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['Comment'])

# Converting text sequences to integer sequences
train_sequences = tokenizer.texts_to_sequences(train_data['Comment'])
val_sequences = tokenizer.texts_to_sequences(val_data['Comment'])
test_sequences = tokenizer.texts_to_sequences(test_data['Comment'])

# Padding sequences to ensure uniform length
max_sequence_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_len)
val_sequences = pad_sequences(val_sequences, maxlen=max_sequence_len)
test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_len)

# Load GloVe embeddings into a dictionary
embeddings_index = {}
embedding_dim = 300  # Change the dimension to match the downloaded file (300d)

with open('C:\\Users\\nikhi\\OneDrive\\Desktop\\Engg Mngt\\SEM 1 - Fall 2023\\EM626 (Applied AI and ML for Sytems and Enterprises)\\Final Project\\glove.42B.300d\\glove.42B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create an embedding matrix for words in your tokenizer
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and len(embedding_vector) == embedding_dim:
        embedding_matrix[i] = embedding_vector
    else:
        # Handling missing or incorrect dimensions by initializing with zeros
        embedding_matrix[i] = np.zeros((embedding_dim,))

# Defining the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
              weights=[embedding_matrix], input_length=max_sequence_len, trainable=False),  # Setting trainable to False
    LSTM(64),
    Dense(6, activation='softmax')
])

# Compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_data['Emotion'], epochs=5, batch_size=64, validation_data=(val_sequences, val_data['Emotion']))

# Evaluate the model and convert predictions to categorical
evaluation = model.evaluate(test_sequences, test_data['Emotion'])
predictions = model.predict(test_sequences)
predictions_categorical = np.argmax(predictions, axis=1)  # Convert predictions to categorical

print(f"Accuracy on test data: {evaluation[1] * 100:.2f}%")

# Creating a confusion matrix
cm = confusion_matrix(test_data['Emotion'], predictions_categorical)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
