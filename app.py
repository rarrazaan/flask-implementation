import cgi
import cgitb
from pyclbr import Class
import pandas as pd
from googletrans import Translator
from sentence_splitter import SentenceSplitter
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import nltk
import csv
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import flask
from flask import flash, redirect, request, url_for, render_template
import os

UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSIONS = {'txt'}

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class Rephrase:
    def __init__(self):
        self.name = 'pegasus_paraphrase'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_tokenizer = PegasusTokenizer.from_pretrained(self.name)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.name).to(self.device)

    # setting up the model
    def paraphrase_sentence(self, sentence, num_return_sequences):
        seq_batch = self.model_tokenizer.prepare_seq2seq_batch(
            [sentence],
            truncation=True,
            padding='longest',
            max_length=500,
            return_tensors="pt"
        ).to(self.device)

        generated_sequence = self.model.generate(**seq_batch,
                                                 max_length=500,
                                                 num_beams=10,
                                                 num_return_sequences=num_return_sequences,
                                                 temperature=0.8)

        paraphrased_sentence = self.model_tokenizer.batch_decode(
            generated_sequence, skip_special_tokens=True)

        return paraphrased_sentence

    splitter = SentenceSplitter(language='en')

    # defining the paraphrase function
    def paraphrase(self, text):
        sentence_list = self.splitter.split(text)
        paraphrased_list = []

        print(sentence_list)

        for sentence in sentence_list:
            a = self.paraphrase_sentence(sentence, 10)
            paraphrased_list.append(a)

        paraphrased_sentences = [' '.join(x) for x in paraphrased_list]
        paraphrased_text = [' '.join(x for x in paraphrased_sentences)]
        paraphrased_text = str(paraphrased_text).strip("\"'[]")

        return paraphrased_text


class T1RNN:
    nltk.download('punkt')
    nltk.download('stopwords')

    def LSA(self, file):
        f = open(file, 'r', encoding="utf8")
        DOCUMENT = f.read()
        DOCUMENT = re.sub(r'(?<=[.,])(?=[^\s])', r' ', str(DOCUMENT))
        DOCUMENT = re.sub(r'\n|\r', ' ', str(DOCUMENT))
        DOCUMENT = re.sub(r' +', ' ', str(DOCUMENT))
        DOCUMENT = DOCUMENT.strip()

        # Tahap memisahkan kalimat yang ada di dokumen
        # pisah perkalimat pake koma
        sentences = nltk.sent_tokenize(str(DOCUMENT))
        # print(sentences)

        # Memasukkan fungsi stopwords ke variabel stop_words
        stop_words = nltk.corpus.stopwords.words('english')

        # Menormalisasi dokumen
        def normalize_document(DOCUMENT):
            # lower case and remove special characters\whitespaces
            isi_document = re.sub(r'[^a-zA-Z\s]', '', DOCUMENT, re.I | re.A)
            isi_document = isi_document.lower()
            isi_document = isi_document.strip()
            # tokenize document
            tokens = nltk.word_tokenize(isi_document)
            # Menyaring stopwords yang ada di dokumen
            filtered_tokens = [
                token for token in tokens if token not in stop_words]
            # Membuat kembali dokumen dari hasil normalisasi dokumen
            DOCUMENT = ' '.join(filtered_tokens)
            return DOCUMENT

        # Mengambil kata yang penting di dalam kalimat
        normalize_corpus = np.vectorize(normalize_document)

        # Mengubah kata dan kalimat penting tadi ke dalam matrix
        normalize_sentences = normalize_corpus(sentences)
        normalize_sentences[:3]

        # Menghitung seberapa sering munculnya kata di dalam kalimat
        tfidVector = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
        dt_matrix = tfidVector.fit_transform(normalize_sentences)
        dt_matrix = dt_matrix.toarray()
        vocab = tfidVector.get_feature_names()
        td_matrix = dt_matrix.T
        # print(td_matrix.shape)
        pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

        # Mengurai matrix dalam bentuk yang sederhana untuk mempermudah pengolahan data
        def low_rank_svd(matrix, singular_count=2):
            u, s, vt = svds(matrix, k=singular_count)
            return u, s, vt

        # diambil 30% dari total kalimat
        number_sentences = round(len(sentences) * 0.3)
        number_topics = 3  # ini k nya

        u, s, vt = low_rank_svd(td_matrix, singular_count=number_topics)
        # print(u.shape, s.shape, vt.shape)
        term_topic_mat, singular_values, topic_document_mat = u, s, vt

        # Menghilangkan Nilai Singular di bawah ambang batas
        sv_threshold = 0.5
        min_sigma_value = max(singular_values) * sv_threshold
        singular_values[singular_values < min_sigma_value] = 0

        salience_scores = np.sqrt(np.dot(np.square(singular_values),
                                         np.square(topic_document_mat)))
        salience_scores

        top_sentence_indices = (-salience_scores).argsort()[:number_sentences]
        top_sentence_indices.sort()

        words = ' '.join(np.array(sentences)[top_sentence_indices])
        #print(words.replace("\\", ""))

        return words.replace("\\", "")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["POST", "GET"])
def index():
    title = "Upload a txt"
    return render_template('index.html', data='', title=title)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    if flask.request.method == "POST":
        translator = Translator()
        translation = translator.translate(request.form['tts'])
        with open("static/news.txt", "w") as text_file:
            text_file.write(translation.text)

        rephrase = Rephrase()
        t1rnn = T1RNN()
        text_summ = t1rnn.LSA('static/news.txt')
        # paraphrase_text = rephrase.paraphrase(text_summ)
        final_summ = translator.translate(text_summ, dest='id')
        return render_template('index.html', data=final_summ)
    return render_template('index.html', data="")


if __name__ == "__main__":
    print(("strating..."))
    app.run(debug=True)
