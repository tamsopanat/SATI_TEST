from flask import Flask
#Model and input data preparation
import fasttext
import pandas as pd
model = fasttext.load_model("./src/ICD10.bin")

#text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

#text preprocessing
def prepare_text(df):
    df['prepare_term'] = df['term'].str.lower()
    df['tokens'] = df['prepare_term'].apply(nltk.word_tokenize)
    df['no_punct_text'] = df['prepare_term'].str.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    df['filtered_text'] = df['no_punct_text'].apply(lambda x: " ".join(word for word in x.split() if word.lower() not in stop_words))
    stemmer = PorterStemmer()
    df['stemmed_text'] = df['filtered_text'].apply(lambda x: " ".join(stemmer.stem(word) for word in x.split()))
    lemmatizer = WordNetLemmatizer()
    df['lemmatized_text'] = df['stemmed_text'].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))
    new_df = df[['lemmatized_text']].copy()
    return new_df

app = Flask(__name__)
app.config['DEBUG'] = True
 
@app.route("/")
def index():
    input_text = 'MRSA' #<--- Input term text
    text_prepared = prepare_text(pd.DataFrame({'term':[input_text]}))
    predict = model.predict(text_prepared.iloc[0,0])
    return predict[0][0].strip('__label__')
 
if __name__ == '__main__':
    app.run(debug = True)