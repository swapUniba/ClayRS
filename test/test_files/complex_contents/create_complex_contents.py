import clayrs.content_analyzer as ca

items_json = "../movies_info_reduced.json"
users_dat = "../users_70.dat"


def item_fit():
    config = ca.ItemAnalyzerConfig(
        ca.JSONFile(items_json),
        id='imdbID',
        output_directory='movies_codified/',
        export_json=True
    )

    config.add_multiple_config(
        'Plot',
        [
            ca.FieldConfig(ca.SkLearnTfIdf(),
                           ca.NLTK(stopwords_removal=True), id='tfidf'),
            ca.FieldConfig(ca.SentenceEmbeddingTechnique(ca.Sbert('paraphrase-distilroberta-base-v1')),
                           ca.NLTK(stopwords_removal=True), id='embedding'),
            ca.FieldConfig(ca.OriginalData(), id='index_original', memory_interface=ca.SearchIndex('index')),
            ca.FieldConfig(ca.OriginalData(), ca.NLTK(stopwords_removal=True),
                           id='index_preprocessed', memory_interface=ca.SearchIndex('index')),
        ]
    )

    config.add_multiple_config(
        'Genre',
        [
            ca.FieldConfig(ca.WordEmbeddingTechnique(ca.Gensim('glove-twitter-25')),
                           ca.NLTK(stemming=True), id='embedding'),
            ca.FieldConfig(ca.WhooshTfIdf(),
                           ca.NLTK(stemming=True), id='tfidf'),
            ca.FieldConfig(ca.OriginalData(), id='index_original', memory_interface=ca.SearchIndex('index')),
            ca.FieldConfig(ca.OriginalData(), ca.NLTK(stopwords_removal=True),
                           memory_interface=ca.SearchIndex('index')),
        ]
    )

    config.add_multiple_config(
        'Year',
        [
            ca.FieldConfig(ca.OriginalData(), id='default_string'),
            ca.FieldConfig(ca.OriginalData(dtype=int), id='int')
        ]
    )

    config.add_single_config(
        'imdbRating',
        ca.FieldConfig(ca.OriginalData(dtype=float))
    )

    config.add_single_exogenous(
        ca.ExogenousConfig(ca.DBPediaMappingTechnique("dbo:Film", "Title"), id='dbpedia')
    )

    ca.ContentAnalyzer(config).fit()


def users_fit():
    config = ca.UserAnalyzerConfig(
        ca.DATFile(users_dat),
        id='0',
        output_directory='users_codified',
        export_json=True
    )

    config.add_single_exogenous(
        ca.ExogenousConfig(ca.PropertiesFromDataset(), id='local')
    )

    ca.ContentAnalyzer(config).fit()


if __name__ == "__main__":
    item_fit()
    users_fit()
