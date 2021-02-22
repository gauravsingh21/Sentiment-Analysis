import emoji
import nltk
from gingerit.gingerit import GingerIt
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import traceback
from textblob import TextBlob
from textblob import Word
import spacy
from tqdm import tqdm

import warnings,os
warnings.filterwarnings('ignore')
class SentimentAnaylsisUtility:
 CONTRACTIONS = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
                 "'cause": "because", "could've": "could have", "couldn't": "could not",
                 "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he he will have",
                 "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                 "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                 "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                 "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                 "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                 "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                 "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                 "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                 "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                 "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                 "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                 "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                 "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                 "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                 "there'd've": "there would have", "there's": "there is", "they'd": "they would",
                 "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                 "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                 "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                 "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                 "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                 "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                 "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                 "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                 "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                 "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                 "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                 "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                 "you're": "you are", "you've": "you have"}
 SMILEY = {"^_^": "smiley", ":‑)": "smiley", ":-]": "smiley", ":-3": "smiley", ":->": "smiley", "8-)": "smiley",
           ":-}": "smiley", ":)": "smiley", ":]": "smiley", ":3": "smiley", ":>": "smiley", "8)": "smiley",
           ":}": "smiley", ":o)": "smiley", ":c)": "smiley", ":^)": "smiley", "=]": "smiley", "=)": "smiley",
           ":-))": "smiley", ":‑D": "smiley", "8‑D": "smiley", "x‑D": "smiley", "X‑D": "smiley", ":D": "smiley",
           "8D": "smiley", "xD": "smiley", "XD": "smiley", ":‑(": "sad", ":‑c": "sad", ":‑<": "sad", ":‑[": "sad",
           ":(": "sad", ":c": "sad", ":<": "sad", ":[": "sad", ":-||": "sad", ">:[": "sad", ":{": "sad", ":@": "sad",
           ">:(": "sad", ":'‑(": "sad", ":'(": "sad", ":‑P": "playful", "X‑P": "playful", "x‑p": "playful",
           ":‑p": "playful", ":‑Þ": "playful", ":‑þ": "playful", ":‑b": "playful", ":P": "playful", "XP": "playful",
           "xp": "playful", ":p": "playful", ":Þ": "playful", ":þ": "playful", ":b": "playful", "<3": "love"}
 STOP_WORDS = {"ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out",
               "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into",
               "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the",
               "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me",
               "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both",
               "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and",
               "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over",
               "why", "so", "can", "did", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only",
               "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against",
               "a", "by", "doing", "it", "how", "further", "was", "here", "than"}
 TAG_DICTIONARY = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

 def __init__(self):
     self.lemmatizer = WordNetLemmatizer()
     self.spell = SpellChecker()
     self.parser = GingerIt()
     plt.figure(figsize=(16, 7))
     plt.style.use('ggplot')
     self.nlp = spacy.load("en_core_web_sm")
     self.all_stopwords = self.nlp.Defaults.stop_words

 def pre_processing(self, text):
    text=text.replace('\\n', '')
    text=text.replace('\'s', '+s')
    text=text.replace('\\', '')
    reformed = [self.CONTRACTIONS[word] if word in self.CONTRACTIONS else word for word in text.split()]
    text = " ".join(reformed)
    text = emoji.demojize(text)
    reformed = [self.SMILEY[word] if word in self.SMILEY else word for word in text.split()]
    text = " ".join(reformed)
    text = ' '.join(re.sub('[^A-Za-z0-9 _+-,.]+', '', text).split())
    return text

 def avg_word(self, sentence):
     words = sentence.split()
     return (sum(len(word) for word in words) / len(words))

 def getDocumentSentimentList(self, docs, splitStr='__label__'):
     docSentimentList = []
     for i in range(len(docs)):
         try:
             text = str(docs[i])
             splitText = text.split(splitStr)
             secHalf = splitText[1]
             text = secHalf[2:len(secHalf) - 1]
             sentiment = secHalf[0]
             docSentimentList.append([i+1, text, sentiment])
         except:
             print("error");

     return docSentimentList;

 def check_grammar(self,text):
     try:
         # text=parser.parse(text)['result']
         text = self.spell.correction(text)
     except Exception:
         traceback.print_exc()

     return text

 def pos_tagging(tokenized):
     wordsList = nltk.word_tokenize(tokenized)
     return nltk.pos_tag(wordsList)

 def lemmatize(self, wordList):
     lemmatizedString = ''
     for word in wordList:
         try:
             text = word[0];
             tag = self.TAG_DICTIONARY.get(word[1][0].upper(), wordnet.NOUN);
             lemmatizedString = lemmatizedString + " " + Word(text).lemmatize(tag)
         except:
             print("error")
             lemmatizedString = lemmatizedString;
     lemmatizedString = lemmatizedString.replace('.', ' ')
     lemmatizedString = lemmatizedString.replace(',', ' ')
     return lemmatizedString.strip();

 def moreCompactKeyword(self, review):
     cleaned = list()
     review = TextBlob(review)
     for phrase in review.noun_phrases:
         count = 0
         for word in phrase.split():
             # Count the number of small words and words without an English definition
             if len(word) <= 2 or (not Word(word).definitions):
                 count += 1
         # Only if the 'nonsensical' or short words DO NOT make up more than 40% (arbitrary) of the phrase add
         # it to the cleaned list, effectively pruning the ones not added.
         if count < len(phrase.split()) * 0.4:
             cleaned.append(phrase)

     for phrase in cleaned:
         match = list()
         temp = list()
         word_match = list()
         for word in phrase.split():
             # Find common words among all phrases
             word_match = [p for p in cleaned if re.search(word, p) and p not in word_match]
             # If the size of matched phrases set is smaller than 30% of the cleaned phrases,
             # then consider the phrase as non-redundant.
             if len(word_match) <= len(cleaned) * 0.3:
                 temp.append(word)
                 match += word_match

         phrase = ' '.join(temp)
         #     print("Match for " + phrase + ": " + str(match))

         if len(match) >= len(cleaned) * 0.1:
             # Redundant feature set, since it contains more than 10% of the number of phrases.
             # Prune all matched features.
             for feature in match:
                 if feature in cleaned:
                     cleaned.remove(feature)

             # Add largest length phrase as feature
             cleaned.append(max(match, key=len))
     return cleaned

 def compactKeyword(self, review):
     cleaned = list()
     review = TextBlob(review)
     for phrase in review.noun_phrases:
         count = 0
         for word in phrase.split():
             # Count the number of small words and words without an English definition
             if len(word) <= 2 or (not Word(word).definitions):
                 count += 1
         # Only if the 'nonsensical' or short words DO NOT make up more than 40% (arbitrary) of the phrase add
         # it to the cleaned list, effectively pruning the ones not added.
         if count < len(phrase.split()) * 0.4:
             containsAdjectiveAdverb = False
             tokens = self.nlp(str(phrase))
             for token in tokens:
                 if (str(token.tag_).startswith("J") or str(token.tag_).startswith("R")):
                     containsAdjectiveAdverb = True;
                     break;
             if (containsAdjectiveAdverb):
                 cleaned.append(phrase)
     return cleaned

 def fix_punctuation(self, text):
     text = text.replace('\\n', '')
     text = text.replace('\'s', '+s')
     text = text.replace('\\', '')
     text = text.replace('.', '. ')
     text = text.replace(',', ', ')
     text = re.sub(' +', ' ', text)
     return text;

 def spacy_pre_processing(self, text):
     cleaned_text = list()
     text = text.lower()
     text = " ".join([self.CONTRACTIONS[word] if word in self.CONTRACTIONS else word for word in text.split()])
     text = emoji.demojize(text)
     text = " ".join([self.SMILEY[word] if word in self.SMILEY else word for word in text.split()])
     text = ' '.join(re.sub('[^A-Za-z0-9 +]+', ' ', text).split())
     text = " ".join([w for w in text.split() if not w in self.STOP_WORDS])
     return text

 def keyword_score(keywords):
     score = 0.0;
     for keyword in keywords:
         score = score + TextBlob(keyword).sentiment.polarity
     return score

 def remove_URL(self, text):
     url = re.compile(r"https?://\S+|www\.\S+")
     return url.sub(r"", text)

 def remove_html(self, text):
     html = re.compile(r"<.*?>")
     return html.sub(r"", text)