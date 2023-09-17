import re
from PyPDF2 import PdfReader
import spacy
from spacy import displacy
import textacy
from textacy import extract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
nlp = spacy.load("en_core_web_md")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def extract_text(filepath):
    """
    Extracts Raw Text from PDF using
    Args:
        filepath(str): Path to the PDF to be extracted
    Returns:
        text(str): Extracted text from the PDF
    """
    text=""
    pdf_file = open(filepath, 'rb')
    reader = PdfReader(pdf_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_text = page.extract_text()
        text = text + page_text
    pdf_file.close()
    return text

def tokenize_data(text):
    """
    Tokenizes raw text and converts into spacy doc object
    Args:
        text(str): Raw text extracted from resume
    Returns:
        doc2(obj): Spacy doc object containing tokens
    """
    doc = nlp(text)
    ext_text = ""
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            ext_text= ext_text + " " + str(token)
    doc2 = nlp(ext_text)
    return doc2
    

class KeytermExtractor:
    """
    A class for extracting keyterms from a given text using various algorithms.
    """

    def __init__(self, raw_text: str, top_n_values: int = 20):
        """
        Initialize the KeytermExtractor object.

        Args:
            raw_text (str): The raw input text.
            top_n_values (int): The number of top keyterms to extract.
        """
        self.raw_text = raw_text
        self.text_doc = textacy.make_spacy_doc(
            self.raw_text, lang="en_core_web_md")
        self.top_n_values = top_n_values

    def get_keyterms_based_on_textrank(self):
        """
        Extract keyterms using the TextRank algorithm.

        Returns:
            List[str]: A list of top keyterms based on TextRank.
        """
        return list(extract.keyterms.textrank(self.text_doc, normalize="lemma",
                                              topn=self.top_n_values))

    def get_keyterms_based_on_sgrank(self):
        """
        Extract keyterms using the SGRank algorithm.

        Returns:
            List[str]: A list of top keyterms based on SGRank.
        """
        return list(extract.keyterms.sgrank(self.text_doc, normalize="lemma",
                                            topn=self.top_n_values))

    def get_keyterms_based_on_scake(self):
        """
        Extract keyterms using the sCAKE algorithm.

        Returns:
            List[str]: A list of top keyterms based on sCAKE.
        """
        return list(extract.keyterms.scake(self.text_doc, normalize="lemma",
                                           topn=self.top_n_values))

    def get_keyterms_based_on_yake(self):
        """
        Extract keyterms using the YAKE algorithm.

        Returns:
            List[str]: A list of top keyterms based on YAKE.
        """
        return list(extract.keyterms.yake(self.text_doc, normalize="lemma",
                                          topn=self.top_n_values))

    def bi_gramchunker(self):
        """
        Chunk the text into bigrams.

        Returns:
            List[str]: A list of bigrams.
        """
        return list(textacy.extract.basics.ngrams(self.text_doc, n=2, filter_stops=True,
                                                  filter_nums=True, filter_punct=True))

    def tri_gramchunker(self):
        """
        Chunk the text into trigrams.

        Returns:
            List[str]: A list of trigrams.
        """
        return list(textacy.extract.basics.ngrams(self.text_doc, n=3, filter_stops=True,
                                                  filter_nums=True, filter_punct=True))

class TextCleaner:

    def __init__(self, raw_text):
        self.stopwords_set = set(stopwords.words(
            'english') + list(string.punctuation))
        self.lemmatizer = WordNetLemmatizer()
        self.raw_input_text = raw_text

    def clean_text(self) -> str:
        """
        Cleans text by removing stopwords and lemmatizing text using nltk
        Args:
            raw_text(str): Text to be cleaned
        Returns:
            cleaned_text(str): Cleaned text
        """
        tokens = word_tokenize(self.raw_input_text.lower())
        tokens = [token for token in tokens if token not in self.stopwords_set]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = ' '.join(tokens)
        return cleaned_text

