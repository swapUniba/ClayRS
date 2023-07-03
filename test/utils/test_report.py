import os.path
import shutil
import unittest
import yaml

from clayrs.content_analyzer import ItemAnalyzerConfig, JSONFile, ExogenousConfig, PropertiesFromDataset, \
    SkLearnTfIdf, NLTK, SentenceEmbeddingTechnique, Spacy, Sbert, WordEmbeddingTechnique, Gensim, FieldConfig, \
    ContentAnalyzer, Ekphrasis, Ratings, CSVFile, Rank
from clayrs.evaluation import EvalModel, Precision, PrecisionAtK, NDCG, MRR, FMeasureAtK
from clayrs.recsys import HoldOutPartitioning, NXPageRank, NXFullGraph, GraphBasedRS
from clayrs.utils.const import datasets_path
from clayrs.utils.report import Report

items_info = os.path.join(datasets_path, 'ml-100k', 'items_info.json')
rat_path = os.path.join(datasets_path, 'ml-100k', 'ratings.csv')


# Obviously it's impossible to test all methods, here we test a bunch
# but manual testing is required for the Report module
class TestReport(unittest.TestCase):

    @staticmethod
    def _build_ca_report():
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(items_info),
            id='movielens_id',
            output_directory='movies_codified/'
        )

        movies_ca_config.add_single_exogenous(ExogenousConfig(PropertiesFromDataset(field_name_list=['review'])))

        movies_ca_config.add_single_config(
            'plot',
            FieldConfig(content_technique=SkLearnTfIdf(),

                        preprocessing=[NLTK(stopwords_removal=True), Ekphrasis()])
        )

        movies_ca_config.add_multiple_config(
            'genres',
            [
                FieldConfig(content_technique=WordEmbeddingTechnique(Gensim()),
                            preprocessing=Spacy(stopwords_removal=True)),

                FieldConfig(content_technique=SentenceEmbeddingTechnique(Sbert()),
                            preprocessing=Spacy(stopwords_removal=True, remove_punctuation=True)),
            ]
        )

        c_analyzer = ContentAnalyzer(movies_ca_config)

        Report(output_dir='test_report').yaml(content_analyzer=c_analyzer)

    def test_ca_yaml(self):

        self._build_ca_report()

        self.assertTrue(os.path.isfile('test_report/ca_report.yml'))

        with open('test_report/ca_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        # Check that exogenous representation codified is present
        exogenous_representations_dict = result.get('exogenous_representations')
        self.assertIsNotNone(exogenous_representations_dict)
        self.assertIsNotNone(exogenous_representations_dict.get('PropertiesFromDataset'))

        # Check that field representations codified are present
        field_dict = result.get("field_representations")
        self.assertIsNotNone(field_dict)

        # Representations for plot field
        plot_dict = field_dict.get('plot/0')
        self.assertIsNotNone(plot_dict)

        self.assertIsNotNone(plot_dict.get('SkLearnTfIdf'))

        plot_preprocessing_dict = plot_dict.get('preprocessing')
        self.assertIsNotNone(plot_preprocessing_dict)
        self.assertIsNotNone(plot_preprocessing_dict.get('NLTK'))
        self.assertIsNotNone(plot_preprocessing_dict.get('Ekphrasis'))

        # Representation 0 for genres field
        genres_0_dict = field_dict.get('genres/0')
        self.assertIsNotNone(genres_0_dict)

        self.assertIsNotNone(genres_0_dict.get('WordEmbeddingTechnique'))

        genres_0_preprocessing_dict = genres_0_dict.get('preprocessing')
        self.assertIsNotNone(genres_0_preprocessing_dict)
        self.assertIsNotNone(genres_0_preprocessing_dict.get('Spacy'))

        # Representation 1 for genres field
        genres_1_dict = field_dict.get('genres/1')
        self.assertIsNotNone(genres_1_dict)

        self.assertIsNotNone(genres_1_dict.get('SentenceEmbeddingTechnique'))

        genres_1_preprocessing_dict = genres_1_dict.get('preprocessing')
        self.assertIsNotNone(genres_1_preprocessing_dict)
        self.assertIsNotNone(genres_1_preprocessing_dict.get('Spacy'))

    @staticmethod
    def _build_rs_report():
        original_rat = Ratings(CSVFile(rat_path))

        pt = HoldOutPartitioning()

        train_list, test_list = pt.split_all(original_rat)

        alg = NXPageRank()

        graph = NXFullGraph(train_list[0])

        gbrs = GraphBasedRS(alg, graph=graph)
        gbrs.rank(test_list[0])

        Report(output_dir='test_report/only_ratings').yaml(original_ratings=original_rat)
        Report(output_dir='test_report/only_partitioning').yaml(partitioning_technique=pt)
        Report(output_dir='test_report/only_recsys').yaml(recsys=gbrs)
        Report(output_dir='test_report/all_rs_module').yaml(original_ratings=original_rat,
                                                            partitioning_technique=pt,
                                                            recsys=gbrs)

    def test_rs_yaml(self):
        self._build_rs_report()

        self.assertTrue(os.path.isfile('test_report/only_ratings/rs_report.yml'))
        self.assertTrue(os.path.isfile('test_report/only_partitioning/rs_report.yml'))
        self.assertTrue(os.path.isfile('test_report/only_recsys/rs_report.yml'))
        self.assertTrue(os.path.isfile('test_report/all_rs_module/rs_report.yml'))

        # test only ratings report
        with open('test_report/only_ratings/rs_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        self.assertTrue(len(result) == 1)
        self.assertIsNotNone(result.get('interactions'))

        # test only partitioning report
        with open('test_report/only_partitioning/rs_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        self.assertTrue(len(result) == 1)
        partitioning_dict = result.get('partitioning')
        self.assertIsNotNone(partitioning_dict)
        self.assertIsNotNone(partitioning_dict.get('HoldOutPartitioning'))

        # test only recsys report
        with open('test_report/only_recsys/rs_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        self.assertTrue(len(result) == 1)
        recsys_dict = result.get('recsys')
        self.assertIsNotNone(recsys_dict)
        self.assertIsNotNone(recsys_dict.get('GraphBasedRS'))

        # test all rs module report
        with open('test_report/all_rs_module/rs_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        self.assertTrue(len(result) == 3)

        self.assertIsNotNone(result.get('interactions'))

        partitioning_dict = result.get('partitioning')
        self.assertIsNotNone(partitioning_dict)
        self.assertIsNotNone(partitioning_dict.get('HoldOutPartitioning'))

        recsys_dict = result.get('recsys')
        self.assertIsNotNone(recsys_dict)
        self.assertIsNotNone(recsys_dict.get('GraphBasedRS'))

    @staticmethod
    def _build_eva_report():

        # rank same as truth since we don't care about results
        rank = Rank(CSVFile(rat_path))
        truth = Ratings(CSVFile(rat_path))

        em = EvalModel([rank],
                       [truth],
                       metric_list=[Precision(),
                                    PrecisionAtK(k=2),
                                    NDCG(),
                                    MRR(),
                                    FMeasureAtK(k=1)])
        em.fit()

        Report('test_report').yaml(eval_model=em)

    def test_eva_yaml(self):
        self._build_eva_report()

        self.assertTrue(os.path.isfile('test_report/eva_report.yml'))

        with open('test_report/eva_report.yml', "r") as stream:
            result = yaml.safe_load(stream)

        self.assertIsNotNone(result.get('n_split'))
        self.assertTrue(result.get('n_split') == 1)

        metrics_dict = result.get('metrics')
        self.assertIsNotNone(metrics_dict)
        self.assertIsNotNone(metrics_dict.get('Precision'))
        self.assertIsNotNone(metrics_dict.get('PrecisionAtK'))
        self.assertIsNotNone(metrics_dict.get('NDCG'))
        self.assertIsNotNone(metrics_dict.get('MRR'))
        self.assertIsNotNone(metrics_dict.get('FMeasureAtK'))

        sys_results_dict = result.get('sys_results')
        self.assertIsNotNone(sys_results_dict.get('sys - fold1'))
        self.assertIsNotNone(sys_results_dict.get('sys - mean'))

        sys_fold_result = sys_results_dict.get('sys - fold1')
        self.assertIsNotNone(sys_fold_result.get('Precision - macro'))
        self.assertIsNotNone(sys_fold_result.get('Precision@2 - macro'))
        self.assertIsNotNone(sys_fold_result.get('NDCG'))
        self.assertIsNotNone(sys_fold_result.get('MRR'))
        self.assertIsNotNone(sys_fold_result.get('F1@1 - macro'))

    def test_eva_yaml_error(self):
        # rank same as truth since we don't care about results
        rank = Rank(CSVFile(rat_path))
        truth = Ratings(CSVFile(rat_path))

        em = EvalModel([rank],
                       [truth],
                       metric_list=[Precision(),
                                    PrecisionAtK(k=2),
                                    NDCG(),
                                    MRR(),
                                    FMeasureAtK(k=1)])

        # try to build report without calling the fit method first
        with self.assertRaises(ValueError):
            Report('test_report').yaml(eval_model=em)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('test_report')


if __name__ == '__main__':
    unittest.main()
