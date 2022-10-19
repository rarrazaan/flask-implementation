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
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import optim
from tqdm import tqdm
import unicodedata

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


class Preprocessing:
    def benerin_text(self, text):  # Benerin text per list dari nilai kolom (belum digabung)
        text = unicodedata.normalize('NFKD', text)
        text = text.replace('  ', ' ')
        text = text.replace('“', '"')
        text = text.replace('‘', "'")
        text = re.sub(r' - - ', '--', text)
        text = re.sub(r'\s([.|,|?|!|:|;|=|%|$])', r'\1', text)
        text = re.sub(r'([#])\s', r'\1', text)
        # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
        text = re.sub(r'\s([/|–|-])\s', r'\1', text)

        # Bakal nyari semua string yang ada di dalem "tanda" dan ngehasilin list dari stringnya
        text_f = re.findall(r'\(.*?\)', text)
        for i in text_f:
            text = text.replace(i, f"({i[2:-2]})")

        text_g = re.findall(r'\".*?\"', text)
        for j in text_g:
            text = text.replace(j, f'"{j[2:-2]}"')

        text_h = re.findall(r'\'.*?\'', text)
        for k in text_h:
            text = text.replace(k, f"'{k[2:-2]}'")

        text_i = re.findall(r'\[.*?\]', text)
        for h in text_i:
            text = text.replace(h, f"[{h[2:-2]}]")

        return text.replace('  ', ' ')

    def preprocess(self, df):
        for ix in range(len(df)):
            text = []
            for i in df['paragraphs'][ix]:
                for j in i:
                    text.append(' '.join(j))
                df['paragraphs'][ix] = ' '.join(text)

        for ix in range(len(df)):
            text = []
            for i in df['summary'][ix]:
                text.append(' '.join(i))
            df['summary'][ix] = ' '.join(text)

        # Apply benerin_text ke column paragh per valuenya
        df['paragraphs'] = df['paragraphs'].apply(
            lambda x: self.benerin_text(x))
        df['summary'] = df['summary'].apply(lambda x: self.benerin_text(x))

        return df


tokenizer = T5Tokenizer.from_pretrained(
    "t5-base-indonesian-summarization-cased")
model = T5ForConditionalGeneration.from_pretrained(
    "t5-base-indonesian-summarization-cased")


class BertDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.summary = self.data["summary"]
        self.paragraphs = self.data["paragraphs"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, index):
        paragraphs = str(self.paragraphs[index])
        summary = str(self.summary[index])
        pad = tokenizer.pad_token
        eos = tokenizer.eos_token
        encoding_paragraphs = self.tokenizer.encode_plus("summarize: " + paragraphs + eos,
                                                         return_token_type_ids=False,
                                                         return_attention_mask=True,
                                                         max_length=512,
                                                         pad_to_max_length=True,
                                                         return_tensors='pt')

        encoding_summary = self.tokenizer.encode(pad + summary + eos,
                                                 add_special_tokens=False,
                                                 return_token_type_ids=False,
                                                 max_length=150,
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
        return {
            'sentence_text': paragraphs,
            'summary_text': summary,
            'input_ids': encoding_paragraphs['input_ids'].flatten(),
            'attention_mask': encoding_paragraphs['attention_mask'].flatten(),
            'lm_labels': encoding_summary.flatten(),
        }


class T5:
    def train(self, train, dev):
        train = Preprocessing.preprocess(train)
        dev = Preprocessing.preprocess(dev)

        train_set = BertDataset(train, tokenizer)
        train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
        val_set = BertDataset(dev, tokenizer)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

        optimizer = optim.AdamW(model.parameters(), lr=3e-5)
        model = model.to("cuda")

        best_val_loss = 999999
        early_stop = 0
        epochs = 100
        for _ in range(epochs):
            model.train()
            train_loss = 0
            for idx, data in tqdm(enumerate(train_loader)):
                sentence_text, summary_text, input_ids, attention_mask, lm_labels = data["sentence_text"], data[
                    "summary_text"], data["input_ids"], data["attention_mask"], data["lm_labels"]
                input_ids = input_ids.to()
                attention_mask = attention_mask.to("cuda")
                lm_labels = lm_labels.to("cuda")
                optimizer.zero_grad()
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask, labels=lm_labels)
                loss, prediction_scores = output[:2]
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                if ((idx % 1000) == 0):
                    print("loss: ", loss.item(),
                          " train_loss: ", train_loss/(idx+1))

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for idx, data in tqdm(enumerate(val_loader)):
                    sentence_text, summary_text, input_ids, attention_mask, lm_labels = data["sentence_text"], data[
                        "summary_text"], data["input_ids"], data["attention_mask"], data["lm_labels"]
                    input_ids = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                    lm_labels = lm_labels.to("cuda")
                    optimizer.zero_grad()
                    output = model(
                        input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels)
                    loss, prediction_scores = output[:2]
                    val_loss += loss.item()

            if ((val_loss/len(val_loader)) < best_val_loss):
                model.save_pretrained("drive/My Drive/model_summarization/")
                best_val_loss = (val_loss/len(val_loader))
            else:
                early_stop += 1
            print("train_loss: ", train_loss/len(train_loader))
            print("val_loss: ", val_loss/len(val_loader))

            if (early_stop == 3):
                break

    def predict(self, pred):
        pred_set = BertDataset(pred, tokenizer)
        pred_loader = DataLoader(pred_set, batch_size=4, shuffle=True)

        with torch.no_grad():
            data = next(iter(pred_loader))
            sentence_text, summary_text, input_ids, attention_mask, lm_labels = data["sentence_text"], data[
                "summary_text"], data["input_ids"], data["attention_mask"], data["lm_labels"]
            input_ids = input_ids.to()
            attention_mask = attention_mask.to()
            lm_labels = lm_labels.to()
            generated = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       min_length=20,
                                       max_length=80,
                                       num_beams=10,
                                       repetition_penalty=2.5,
                                       length_penalty=1.0,
                                       early_stopping=True,
                                       no_repeat_ngram_size=2,
                                       use_cache=True,
                                       do_sample=True,
                                       temperature=0.8,
                                       top_k=50,
                                       top_p=0.95)
            tokenizer.decode(generated[0])
            # print("full text")
            # print(sentence_text[0])
            # print("summary")
            # print(summary_text[0])
            # print("Generated summary")
            # print(tokenizer.decode(generated[0], skip_special_tokens=True))
            return tokenizer.decode(generated[0], skip_special_tokens=True)


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
        file = request.form['tts']
        with open("static/news.txt", "w") as text_file:
            text_file.write(file)

        pred = pd.read_json('static/news.txt', lines=True)
        rephrase = Rephrase()
        t1rnn = T1RNN()
        t5 = T5()
        text_summ = t5.predict(pred)
        # paraphrase_text = rephrase.paraphrase(text_summ)
        # final_summ = translator.translate(text_summ, dest='id')
        return render_template('index.html', data=text_summ)
    return render_template('index.html', data="")


if __name__ == "__main__":
    print(("strating..."))
    app.run(debug=True)
