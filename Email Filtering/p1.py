import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data_path = "C:\datasets\spam.csv"
email_data = pd.read_csv(data_path, encoding='latin-1')


email_data['sms'] = email_data['sms'].str.lower().str.replace(r'[^\w\s]+', '')
email_data['label'] = email_data['label'].map({'ham': 0, 'spam': 1})


tokenizer = Tokenizer()
tokenizer.fit_on_texts(email_data['sms'])
sequences = tokenizer.texts_to_sequences(email_data['sms'])


max_length = max(len(s) for s in sequences)


data = pad_sequences(sequences, maxlen=max_length)


labels = np.asarray(email_data['label'])
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
lstm_units = 128


model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)


model.save('spam_model.h5')


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
