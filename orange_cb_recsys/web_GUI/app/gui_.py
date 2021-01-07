import os
import zipfile
from typing import Optional, Any

from flask import Flask, render_template, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

import json as js
import yaml
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON

import numpy as np


#  class which parse a dataset selected by the user
class Parser:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    # file json
    def from_file_json(self):
        list_dict_record = []
        list_key = []

        def convertJsonFile():
            with open(self.dataset_path) as js_file:
                for record in js_file:
                    dict_record = js.loads(record)  # prendo ogni record
                    list_dict_record.append(dict_record)  # ogni record preso è inserito in una lista

        def store_key():
            for i in list_dict_record:
                for key in i:
                    list_key.append(key)  # target: prendo ogni key value presente nel dataset e lo passo ad una lista

        def unique_key_value():
            list_key_unique = np.unique(list_key)  # unique list key values
            list_key_unique_ = list_key_unique.tolist()
            return list_key_unique_

        convertJsonFile()
        store_key()
        list_field = unique_key_value()
        return list_field

    # file csv
    def from_file_csv(self):
        data = pd.read_csv(self.dataset_path, nrows=0)
        list_field = []
        for i in data:
            list_field.append(i)

        return list_field

    # file dat
    def from_file_dat(self):
        fields_list = []
        with open(self.dataset_path) as dat_file:
            for line in dat_file:
                fields = line.split('::')
                y = 0
                for _ in fields:
                    fields_list.append('field_pos_' + str(y))
                    y = y + 1
                break
        return fields_list

# function that get value field selected in a dat file
def getValue_dat(field_):
    field_splitted = field_.split('_')
    return str(field_splitted[-1])


#  class which gets all the classes of the DBpedia ontology
def returnTypeDBpedia():
    _types = []

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.addDefaultGraph("http://dbpedia.org")

    query = "select ?type {"
    query += "   ?type a owl:Class ."
    query += "}"

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        type_ = str(result["type"]["value"])
        data = type_.split('/')
        _types.append(data[4])

    return _types


#  set the app FLASK (GUI)
app = Flask(__name__, template_folder="../app/templates")
app.debug = True

UPLOAD_FOLDER = "../app/upload"
ALLOWED_EXTENSIONS = {'json', 'csv', 'dat', 'bin'}


#  this function manages to queue the files (uploaded by the user) in the directory 'upload'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#  data important for the iter wizard of the guy
content_type = ''

dataset = ''

selected_fields = []

content_tech_list = ["embedding",
                     "babelpy",
                     "lucene_tf-idf",
                     "search_index",
                     "sk_learn_tf-idf",
                     "synset_frequency",
                     "dbpedia_mapping"]

rating_tech_list = ["text_blob_sentiment",
                    "number_normalizer"]

embedding_combining_tech_list = ["centroid"]

embidding_granularity = ["word",
                         "doc",
                         "sentence"]

embedding_source_list = ["binary_file",
                         "gensim_downloader"]

binary_type = ["word2vec",
               "doc2vec",
               "fasttext",
               "ri"]

gensim_models = ["conceptnet-numberbatch-17-06-300",
                 "fasttext-wiki-news-subwords-300",
                 "glove-twitter-25",
                 "glove-twitter-50",
                 "glove-twitter-100",
                 "glove-twitter-200",
                 "glove-wiki-gigaword-50",
                 "glove-wiki-gigaword-100",
                 "glove-wiki-gigaword-200",
                 "glove-wiki-gigaword-300",
                 "word2vec-google-news-300",
                 "word2vec-ruscorpora-300"]

preprocessing_tech_list = ["stopwords_removal",
                           "stemming",
                           "lemmatization",
                           "strip_multiple_whitespaces",
                           "url_tagging"]

dbpedia_mode = ["all",
                "all_retrieved",
                "only_retrieved_evaluated",
                "original_retrieved"]


#  config data for the configuration file
config_dict_content = {}
config_dict_rating = {}

@app.route('/')
def homepage():
    return render_template('home.html')

#  help section
@app.route('/help')
def help_():
    return render_template('help.html')

@app.route('/download_help')
def download_help():
    help_file = 'documentation.pdf'
    return send_file(help_file, as_attachment=True)

#

@app.route('/', methods=['GET', 'POST'])
def content_type_selected():
    global content_type
    if request.method == 'POST':
        # field content type
        content_type = request.form.get('content')
        config_dict_content['content_type'] = content_type
        config_dict_rating['content_type'] = content_type

        return redirect(url_for('upload'))


@app.route('/upload_files')
def upload():
    return render_template('upload_file.html')


#  function that views the dataset uploaded by the user in the GUI
def view_dataset(dataset_):
    data = []

    for file in os.listdir(UPLOAD_FOLDER):
        if file == dataset_:
            dataset_path = os.path.join(UPLOAD_FOLDER, file)
            file = open(dataset_path)
            lines = file.readlines()
            for line in lines:
                data.append(line)

    return data


@app.route('/upload_files', methods=['GET', 'POST'])
def upload_selected():
    if request.method == 'POST':
        config_dict_content['output_directory'] = request.form.get('dir')
        config_dict_rating['output_directory'] = request.form.get('dir')

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            global dataset
            dataset = filename

            config_dict_content['raw_source_path'] = "web_GUI/app/upload/{}".format(dataset)
            config_dict_rating['raw_source_path'] = "web_GUI/app/upload/{}".format(dataset)
            if dataset.endswith('.json'):
                config_dict_content['source_type'] = 'json'
                config_dict_rating['source_type'] = 'json'

            if dataset.endswith('.csv'):
                config_dict_content['source_type'] = 'csv'
                config_dict_rating['source_type'] = 'csv'

            if dataset.endswith('.dat'):
                config_dict_content['source_type'] = 'dat'
                config_dict_rating['source_type'] = 'dat'

            return redirect(url_for('select_fields'))


#  function that allows to extract the fields of the selected dataset by the user
def extract_field():
    global dataset
    for file in os.listdir(UPLOAD_FOLDER):
        if file == dataset:
            dataset_path = os.path.join(UPLOAD_FOLDER, file)

            list_field = []
            parser = Parser(os.path.abspath(dataset_path))
            if dataset.endswith('.json'):
                list_field = parser.from_file_json()
            if dataset.endswith('.csv'):
                list_field = parser.from_file_csv()
            if dataset.endswith('.dat'):
                list_field = parser.from_file_dat()

            return list_field
    return None


@app.route('/select_fields')
def select_fields():
    global dataset
    global content_type
    content_type_ = config_dict_content['content_type']

    data = view_dataset(dataset)

    list_field_ = extract_field()

    return render_template('select_fields.html', **locals())


@app.route('/select_fields', methods=['GET', 'POST'])
def selected_fields():
    global selected_fields
    global content_type
    global dataset
    global config_dict_content_n_f
    global config_dict_rating_n_f

    if request.method == 'POST':
        if content_type != 'RATING':
            id_fields_selected = request.form.getlist('id_fields')
            fields_selected = request.form.getlist('fields')

            if dataset.endswith('.dat'):
                id_fields_selected_dat = []
                for field in id_fields_selected:
                    id_fields_selected_dat.append(getValue_dat(field))
                config_dict_content['id_field_name'] = id_fields_selected_dat
            else:
                config_dict_content['id_field_name'] = id_fields_selected

        else:
            fields_selected = request.form.getlist('rating_fields')

            if dataset.endswith('.dat'):
                from_field = request.form.get('from_fields')
                to_field = request.form.get('to_fields')
                timestamp = request.form.get('timestamp_fields')
                config_dict_rating['from_field_name'] = getValue_dat(from_field)
                config_dict_rating['to_field_name'] = getValue_dat(to_field)
                config_dict_rating['timestamp_field_name'] = getValue_dat(timestamp)

            else:
                config_dict_rating['from_field_name'] = request.form.get('from_fields')
                config_dict_rating['to_field_name'] = request.form.get('to_fields')
                config_dict_rating['timestamp_field_name'] = request.form.get('timestamp_fields')

        selected_fields = fields_selected

        if len(selected_fields) != 0:
            return redirect(url_for('technique_'))
        else:
            return redirect(url_for('download_files'))


@app.route('/technique_')
def technique_():
    global content_type, selected_fields, dataset, content_tech_list, rating_tech_list, embedding_combining_tech_list, embidding_granularity, embedding_source_list, binary_type, gensim_models, dbpedia_mode
    content_type_ = config_dict_content['content_type']

    selected_fields_ = selected_fields

    data = view_dataset(dataset)
    # content section
    content_tech_list_ = content_tech_list

    preprocessing_tech_list_ = preprocessing_tech_list
    embedding_combining_tech_list_ = embedding_combining_tech_list
    embidding_granularity_ = embidding_granularity
    embedding_source_list_ = embedding_source_list
    binary_type_ = binary_type
    gensim_models_ = gensim_models

    types = returnTypeDBpedia()
    dbpedia_mode_ = dbpedia_mode

    # rating section
    rating_tech_list_ = rating_tech_list

    return render_template('technique_.html', **locals())


@app.route('/technique_', methods=['GET', 'POST'])
def technique_selected():
    global selected_fields
    global content_type
    global dataset

    fields_list = []

    if request.method == 'POST':

        if content_type != 'RATING':
            config_dict_content['fields'] = fields_list
            exogenus_list = []
            config_dict_content['get_lod_properties'] = exogenus_list
            for field in selected_fields:
                techniques_selected_ = request.form.getlist('{}tech'.format(field))
                techniques_exogenus_selected_ = request.form.getlist('{}tech_exogenus'.format(field))

                if len(techniques_selected_) != 0:
                    field_dict = {}
                    if dataset.endswith('.dat'):
                        field_dict['field_name'] = getValue_dat(field)
                    else:
                        field_dict['field_name'] = field

                    field_dict['lang'] = 'EN'
                    field_dict['memory_interface'] = 'None'
                    field_dict['memory_interface_path'] = 'None'

                    pipeline_list = []
                    field_dict['pipeline_list'] = pipeline_list

                    if 'search_index' in techniques_selected_:
                        config_dict_content['search_index'] = 'True'
                    else:
                        config_dict_content['search_index'] = 'False'

                    for technique in techniques_selected_:
                        tech_dict = {}

                        #  get all the parameters selected by the user for embedding technique
                        if technique == 'embedding':
                            combining_tech_selected = request.form.get('{}{}combining'.format(field, technique))
                            granularity_selected = request.form.get('{}{}granularity'.format(field, technique))

                            source_selected = request.form.get('{}{}source'.format(field, technique))
                            if source_selected == 'gensim_downloader':
                                model_selected = request.form.get('{}{}_models'.format(field, technique))
                                gensim_dict = {"class": source_selected, "name": model_selected}

                                tech_dict['field_content_production'] = {'class': technique, 'combining_technique': {'class': combining_tech_selected},
                                                                         'granularity': granularity_selected, "embedding_source": gensim_dict}

                            if source_selected == 'binary_file':
                                binary_file = request.files['{}{}binary_file'.format(field, technique)]
                                binary_file_selected = binary_file.filename
                                if not binary_file_selected.endswith('.bin'):
                                    return redirect(request.url)

                                type_selected = request.form.get('{}{}{}_type'.format(field, technique, source_selected))
                                binary_dict = {"class": source_selected, "file_path": "web_GUI/app/upload/{}".format(binary_file_selected), "embedding_type": type_selected}

                                tech_dict['field_content_production'] = {'class': technique, 'combining_technique': {'class': combining_tech_selected},
                                                                         'granularity': granularity_selected, "embedding_source": binary_dict}
                        else:
                            tech_dict['field_content_production'] = {'class': technique}

                        #  nltk section
                        preprocessing_list = []
                        tech_dict['preprocessing_list'] = preprocessing_list

                        # get all the nltk techniques selected for every field selected by the use
                        tech_nltk_list = request.form.getlist('{}{}nltk_tech'.format(field, technique))
                        if len(tech_nltk_list) != 0:
                            field_nltk_dict = {"class": "nltk"}
                            for nltk_tech in tech_nltk_list:
                                field_nltk_dict[nltk_tech] = "True"

                            preprocessing_list.append(field_nltk_dict)

                        pipeline_list.append(tech_dict)

                    fields_list.append(field_dict)

                #  exogenus section
                if len(techniques_exogenus_selected_) != 0:
                    for technique_exo in techniques_exogenus_selected_:
                        if technique_exo == 'dbpedia_mapping':
                            db_tech_dict = {'class': technique_exo,
                                            'lang': 'EN',
                                            'entity_type': request.form.get('{}{}entity'.format(field, technique_exo)),
                                            'mode': request.form.get('{}{}mode'.format(field, technique_exo))}
                            if dataset.endswith('.dat'):
                                db_tech_dict['label_field'] = getValue_dat(field)
                            else:
                                db_tech_dict['label_field'] = field

                            exogenus_list.append(db_tech_dict)

        else:
            config_dict_rating['fields'] = fields_list
            for field in selected_fields:
                field_dict = {}
                if dataset.endswith('.dat'):
                    field_dict['field_name'] = getValue_dat(field)
                else:
                    field_dict['field_name'] = field
                techniques_selected_ = request.form.get('{}tech'.format(field))
                if techniques_selected_ == 'number_normalizer':
                    processor_dict = {"class": techniques_selected_,
                                      "min_": float(request.form.get('{}{}min'.format(field, techniques_selected_))),
                                      "max_": float(request.form.get('{}{}max'.format(field, techniques_selected_)))}
                else:
                    processor_dict = {"class": techniques_selected_}

                field_dict["processor"] = processor_dict

                fields_list.append(field_dict)

    return redirect(url_for('download_files'))


@app.route('/download')
def download_files():
    global content_type
    global selected_fields

    config_list = []
    config_list.clear()

    if len(selected_fields) != 0:
        if content_type != 'RATING':
            config_list.append(config_dict_content)
        else:
            config_list.append(config_dict_rating)

    else:
        if content_type != 'RATING':
            config_dict_content.pop('fields', None)
            config_dict_content.pop('get_lod_properties', None)
            config_dict_content.pop('search_index', None)

            config_list.append(config_dict_content)
        else:
            config_list.append(config_dict_rating)

    with open("configuration_files/config.json", 'w+') as json_file:
        js.dump(config_list, json_file, indent=2)

    with open("configuration_files/config.yml", 'w+') as yml_file:
        yaml.dump(config_list, yml_file)

    selected_fields_ = selected_fields

    return render_template('download.html', selected_fields_=selected_fields_)


# Declare the function to return all file paths of the particular directory
def retrieve_file_paths(dirName):
    # setup file paths variable
    filePaths = []

    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(dirName):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)

    # return all paths
    return filePaths

# zip directory
def Zip_Configuration_files(directory):
    filePaths = retrieve_file_paths(directory)
    zip_file = zipfile.ZipFile(directory + '.zip', 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)


@app.route('/download_files')
def download_():
    directory = 'configuration_files'
    Zip_Configuration_files(directory)

    path_zip = 'configuration_files.zip'

    return send_file(path_zip, as_attachment=True)


if __name__ == '__main__':
    app.run(port=8080)  # host='0.0.0.0', port=5001  ___ for docker
