import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from bs4 import BeautifulSoup
import re
import unicodedata
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords                   
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from numpy import load
datacallers = load('uniquecallers.npy',allow_pickle=True)



def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    #Using regex
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    return text

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  

def clean_text(text):
    text=text.lower()
    text= re.sub(r"_x000D_",' ',text)
    text = re.sub(r'[\r|\n|\r\n]+', ' ',text)
    text = re.sub(r"received from:",' ',text)
    text = re.sub(r"from:",' ',text)
    text = re.sub(r"to:",' ',text)
    text = re.sub(r"subject:",' ',text)
    text = re.sub(r"sent:",' ',text)
    text = re.sub(r"ic:",' ',text)
    text = re.sub(r"cc:",' ',text)
    text = re.sub(r"bcc:",' ',text)
    text = re.sub(r"issue resolved.",' ', text)
    # Removing url
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    #Removing email 
    text = re.sub(r'\S+@\S+', '', text)
    text = text.replace("\\", ' ')
    # Removing numbers 
    text = re.sub(r'\d+','' ,text)
    # Removing accented characters
    text = remove_accented_chars(text)
    # Remove new line characters 
    text = re.sub(r'\n',' ',text)
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    #& 
    text = re.sub(r'&;?', 'and',text)
    # Remove HTML special entities (e.g. &amp;)
    text= strip_html_tags(text)
    # Remove characters beyond Readable formart by Unicode:
    text= ''.join(c for c in text if c <= '\uFFFF') 
    text = text.strip()
    # Removing special characters and\or digits    
    special_char_pattern = re.compile(r'([{.(-)!_,}])')
    text = special_char_pattern.sub(" \\1 ", text)
    text = remove_special_characters(text, remove_digits=True) 
    # Remove unreadable characters  (also extra spaces)
    text = ' '.join(re.sub("[^\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    for name in datacallers:
              namelist = [part for part in name.split()]
              for namepart in namelist: 
                      text = text.replace(namepart,'')
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()

    return text
def detectln(text):
    try:
        result_lang = detect(text)
    except:
        result_lang ='Other'
    return result_lang

#simple function to detect and translate text 
def detect_translate(text,target_lang):
    
    result_lang = detect(text)
    
    if result_lang == target_lang:
        return text 
    
    else:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
def lemmatize_corpus(corpus, text_lemmatization=True, stopword_removal=True):
    
    lemmatize_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        lemmatize_corpus.append(doc)
        
    return lemmatize_corpus

# preprocessing data
def preprocess_data(data):
    data.drop(["Caller","Short description"],axis=1,inplace= True)
    data['Clndescription'] = data['Description'].apply(clean_text)
    data["Language"] = data['Clndescription'].apply(lambda x: detectln(x))
    data=data[data["Language"] != 'Other']
    data["Trndescription"] = data['Clndescription'].apply(lambda x: detect_translate(x,target_lang='en'))
    data['lmdescription'] = lemmatize_corpus(data['Trndescription'])
#     vectorizer=TfidfVectorizer(max_df=0.7,analyzer='word')
#     vectorizer.fit(data)
#     data_tf=vectorizer.transform(data)
    # tfidf = pickle.load(open("tfidf1.pkl", "rb" ) )
    with open('tfidf1.pkl','rb') as f:
        tfidf=pickle.load(f)
    # tf1_new = TfidfVectorizer(max_df=0.7,analyzer='word', vocabulary = tfidf.vocabulary_)
    X_tf1 = tfidf.transform(data['lmdescription'])
    with open('svc.pkl','rb') as f:
        svcmod=pickle.load(f)
    pred=svcmod.predict(X_tf1)    
    return pred
    
    
