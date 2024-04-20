import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('spam_model.h5')


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_length = 189


def predict_spam(sentence):

    sentence = sentence.lower().replace(r'[^\w\s]+', '')
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)

    prediction = model.predict(padded_sequence)

    if prediction[0][0] > 0.5:
        return "Spam"
    else:
        return "Not Spam"


# sentence = "Congratulations! You've won a free ticket to Bahamas. Text 'WON' to 12345 now!"
sentence = "Hello, did you send me the files?"
result = predict_spam(sentence)
print(f'The sentence is: {result}')
