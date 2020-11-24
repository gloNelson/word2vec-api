
'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/wor2vec/n_similarity/ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

from flask import Flask
from flask_restful import Resource, Api, reqparse
from gensim.models import KeyedVectors
import nltk
nltk.download('punkt')
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import word_tokenize
import argparse
import pickle
import base64


parser = reqparse.RequestParser()


def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in nlp_model.vocab]


class StanfordTagger(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('s', type=str, required=True, help="Sentence cannot be blank!")
        args = parser.parse_args()
        words = word_tokenize("I love you.")
        print(str(words))
        print(pos_tagger.tag(words))
        return str(pos_tagger.tag(words))

class Contains(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w', type=str, required=True, help="Word cannot be blank!")
        args = parser.parse_args()
        return args['w'] in nlp_model.vocab


class N_Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ws1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        parser.add_argument('ws2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        args = parser.parse_args()
        return nlp_model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2']))


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return float(nlp_model.similarity(args['w1'], args['w2']))


class MostSimilar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print("positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t))
        try:
            res = nlp_model.most_similar_cosmul(positive=pos,negative=neg,topn=t)
            return res
        except Exception as e:
            print(e)
            print(res)


class Model(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="word to query.")
        args = parser.parse_args()
        try:
            res = nlp_model[args['word']]
            res = base64.b64encode(res)
            return res
        except Exception as e:
            print(e)
            return

class ModelWordSet(Resource):
    def get(self):
        try:
            res = base64.b64encode(pickle.dumps(set(nlp_model.index2word)))
            return res
        except Exception as e:
            print(e)
            return

app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global nlp_model
    global pos_tagger
    #----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--w2v_model", help="Path to the trained model")
    p.add_argument("--tag_model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--jar", help="Path to the Stanford POS Tagger .jar file")
    args = p.parse_args()

    host = args.host if args.host else "localhost"
    port = int(args.port) if args.port else 5000
    binary = True if args.binary else False
    w2v_model_path = args.w2v_model if args.w2v_model else "./model.bin.gz"
    nlp_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=binary)
    api.add_resource(N_Similarity, '/word2vec/n_similarity')
    api.add_resource(Similarity, '/word2vec/similarity')
    api.add_resource(MostSimilar, '/word2vec/most_similar')
    api.add_resource(Model, '/word2vec/model')
    api.add_resource(ModelWordSet, '/word2vec/model_word_set')
    api.add_resource(Contains, '/word2vec/contains')

    jar = args.jar if args.jar else "./stanford-postagger/stanford-postagger.jar"
    tag_model_path = args.tag_model if args.tag_model else "./stanford-postagger/models/english-bidirectional-distsim.tagger"
    pos_tagger = StanfordPOSTagger(tag_model_path, jar, encoding = 'utf-8')
    api.add_resource(StanfordTagger, '/stanford_tagger/partOfSpeech')

    app.run(host=host, port=port)
