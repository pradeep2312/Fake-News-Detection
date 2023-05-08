import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import confusion_matrix

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

tokenizer = Tokenizer()


vector_form=pickle.load(open('tokenized_data.pkl','rb'))
load_model=pickle.load(open('Sequential.pkl','rb'))


def predictData(sourceData):
  tokenizer.fit_on_texts(sourceData)
  sourceData = tokenizer.texts_to_sequences(sourceData)
  sourceData = pad_sequences(sourceData, maxlen=1000)
  final_value = (load_model.predict(sourceData)>=0.5).astype(int)

  return final_value




if __name__=='__main__':
    st.title('Fake News Detection  ')
    st.subheader("Give the input News below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    sentence = sentence.split()
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=predictData(sentence)
        print(prediction_class)
        if prediction_class.any() <= 0.5:
            st.success('Fake')
        else:
            st.warning('Real')