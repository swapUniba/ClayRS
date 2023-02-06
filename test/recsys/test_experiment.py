import os.path
import shutil
import unittest

from clayrs.content_analyzer import CSVFile, Ratings
from clayrs.evaluation import Precision, Recall
from clayrs.recsys import HoldOutPartitioning, CentroidVector, CosineSimilarity, SkKNN, ClassifierRecommender, \
    LinearPredictor, SkLinearRegression, SkSVC, KFoldPartitioning, NXPageRank
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.experiment import ContentBasedExperiment, GraphBasedExperiment
from test import dir_test_files

rat_path = os.path.join(dir_test_files, "new_ratings.csv")
items_dir = os.path.join(dir_test_files, "complex_contents", "movies_codified")

# for coverage
num_cpus = 1


class TestExperiment(unittest.TestCase):
    def test_already_existent_output_folder(self):
        rat = Ratings(CSVFile(rat_path))

        # test a subclass since Experiment is abstract
        cbe = ContentBasedExperiment(rat,
                                     partitioning_technique=HoldOutPartitioning(),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity(),
                                                                    threshold=2)],
                                     items_directory=items_dir,
                                     output_folder="my_experiment")

        cbe.rank(num_cpus=num_cpus)

        self.assertTrue(os.path.isdir("my_experiment"))

        with self.assertRaises(FileExistsError):
            # same output folder and overwrite parameter not set
            cbe = ContentBasedExperiment(rat,
                                         partitioning_technique=HoldOutPartitioning(),
                                         algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity(),
                                                                        threshold=2)],
                                         items_directory=items_dir,
                                         output_folder="my_experiment")

            cbe.rank(num_cpus=num_cpus)

        # same output folder and overwrite parameter is set
        cbe = ContentBasedExperiment(rat,
                                     partitioning_technique=HoldOutPartitioning(),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity(),
                                                                    threshold=2)],
                                     items_directory=items_dir,
                                     output_folder="my_experiment",
                                     overwrite_if_exists=True,
                                     report=True)

        cbe.rank(num_cpus=num_cpus)

        self.assertTrue(os.path.isdir("my_experiment"))

        # different output folder
        cbe = ContentBasedExperiment(rat,
                                     partitioning_technique=HoldOutPartitioning(),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity(),
                                                                    threshold=2)],
                                     items_directory=items_dir,
                                     output_folder="my_experiment_2",
                                     overwrite_if_exists=True,
                                     report=True)

        cbe.rank(num_cpus=num_cpus)

        self.assertTrue(os.path.isdir("my_experiment"))
        self.assertTrue(os.path.isdir("my_experiment_2"))

        # clean folders created
        shutil.rmtree("my_experiment", ignore_errors=False)
        shutil.rmtree("my_experiment_2", ignore_errors=False)

    def test_eval_metrics(self):
        rat = Ratings(CSVFile(rat_path))

        # test a subclass since Experiment is abstract
        cbe = ContentBasedExperiment(rat,
                                     partitioning_technique=HoldOutPartitioning(),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     LinearPredictor({'Plot': 0}, regressor=SkLinearRegression())],
                                     items_directory=items_dir,
                                     output_folder="my_experiment",
                                     metric_list=[Precision(), Recall()])

        cbe.predict(num_cpus=num_cpus)

        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "eva_sys_results.csv")))
        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "eva_users_results.csv")))

        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "eva_sys_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "eva_users_results.csv")))

    def test_eval_metrics_with_report(self):
        rat = Ratings(CSVFile(rat_path))

        # test a subclass since Experiment is abstract
        cbe = ContentBasedExperiment(rat,
                                     partitioning_technique=HoldOutPartitioning(),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     LinearPredictor({'Plot': 0}, regressor=SkLinearRegression())],
                                     items_directory=items_dir,
                                     output_folder="my_experiment",
                                     metric_list=[Precision(), Recall()],
                                     report=True)

        cbe.predict(num_cpus=num_cpus)

        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "eva_sys_results.csv")))
        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "eva_users_results.csv")))
        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "eva_report.yml")))

        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "eva_sys_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "eva_users_results.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "eva_report.yml")))

    def doCleanups(self) -> None:
        shutil.rmtree("my_experiment", ignore_errors=True)


class TestContentBasedExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rat = Ratings(CSVFile(rat_path))
        cls.rat = rat

        cls.cbe = ContentBasedExperiment(rat,
                                         partitioning_technique=KFoldPartitioning(random_state=42, n_splits=2),
                                         algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                         ClassifierRecommender({'Plot': 0}, classifier=SkSVC()),
                                                         ClassifierRecommender({'Plot': 0}, classifier=SkKNN(2)),
                                                         LinearPredictor({'Plot': 0}, regressor=SkLinearRegression())],
                                         items_directory=items_dir,
                                         output_folder="my_experiment")

    def test_rank(self):
        self.cbe.rank(num_cpus=num_cpus)

        # check dirs have been created
        self.assertTrue(os.path.isdir("my_experiment"))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "CentroidVector_1")))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "ClassifierRecommender_1")))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "ClassifierRecommender_2")))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "LinearPredictor_1")))

        # check user_map and item_map have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "user_map.yml")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "item_map.yml")))

        # check train test splits have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split1.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split1.csv")))

        # check rank created for centroid vector 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "rs_rank_split1.csv")))

        # check rank created for classifier recommender 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_1", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_1", "rs_rank_split1.csv")))

        # check rank created for classifier recommender 2
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_2", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_2", "rs_rank_split1.csv")))

        # check rank created for linear predictor 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_rank_split1.csv")))

    def test_rank_with_report(self):
        cbe = ContentBasedExperiment(self.rat,
                                     partitioning_technique=KFoldPartitioning(random_state=42, n_splits=2),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkSVC()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkKNN(2)),
                                                     LinearPredictor({'Plot': 0}, regressor=SkLinearRegression())],
                                     items_directory=items_dir,
                                     output_folder="my_experiment",
                                     report=True)

        cbe.rank(num_cpus=num_cpus)

        # check report created for centroid vector 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "CentroidVector_1", "rs_report.yml")))

        # check report created for classifier recommender 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_1", "rs_report.yml")))

        # check report created for classifier recommender 2
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "ClassifierRecommender_2", "rs_report.yml")))

        # check report created for linear predictor 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_report.yml")))

    def test_predict(self):
        self.cbe.predict(num_cpus=num_cpus)

        # check dirs have been created
        self.assertTrue(os.path.isdir("my_experiment"))
        self.assertFalse(os.path.isdir(os.path.join("my_experiment", "CentroidVector_1")))
        self.assertFalse(os.path.isdir(os.path.join("my_experiment", "ClassifierRecommender_1")))
        self.assertFalse(os.path.isdir(os.path.join("my_experiment", "ClassifierRecommender_2")))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "LinearPredictor_1")))

        # check user_map and item_map have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "user_map.yml")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "item_map.yml")))

        # check train test splits have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split1.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split1.csv")))

        # check predict created for linear predictor
        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_rank_split0.csv")))
        self.assertFalse(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_rank_split1.csv")))

    def test_predict_with_report(self):
        cbe = ContentBasedExperiment(self.rat,
                                     partitioning_technique=KFoldPartitioning(random_state=42, n_splits=2),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkSVC()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkKNN(2)),
                                                     LinearPredictor({'Plot': 0}, regressor=SkLinearRegression())],
                                     items_directory=items_dir,
                                     output_folder="my_experiment",
                                     report=True)

        cbe.predict(num_cpus=num_cpus)

        # check report created for linear predictor 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "LinearPredictor_1", "rs_report.yml")))

    def test_no_one_predictor(self):
        cbe = ContentBasedExperiment(self.rat,
                                     partitioning_technique=HoldOutPartitioning(random_state=42),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkSVC()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkKNN(2))],
                                     items_directory=items_dir,
                                     output_folder="my_experiment")

        cbe.predict(num_cpus=num_cpus)

        self.assertTrue(os.path.isdir("my_experiment"))

        # since no algorithm is a score prediction one, only train, test split and item and user map will be present
        # and no other thing
        contents_directory = os.listdir("my_experiment")

        contents_directory.remove("HoldOutPartitioning_train_split0.csv")
        contents_directory.remove("HoldOutPartitioning_test_split0.csv")
        contents_directory.remove("user_map.yml")
        contents_directory.remove("item_map.yml")

        self.assertTrue(len(contents_directory) == 0)

    def test_predict_raise_error(self):
        cbe = ContentBasedExperiment(self.rat,
                                     partitioning_technique=HoldOutPartitioning(random_state=42),
                                     algorithm_list=[CentroidVector({'Plot': 0}, similarity=CosineSimilarity()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkSVC()),
                                                     ClassifierRecommender({'Plot': 0}, classifier=SkKNN(2))],
                                     items_directory=items_dir,
                                     output_folder="my_experiment")

        with self.assertRaises(NotPredictionAlg):
            cbe.predict(num_cpus=num_cpus, skip_alg_error=False)

    def doCleanups(self) -> None:
        # clean folders created
        shutil.rmtree("my_experiment", ignore_errors=True)


class TestGraphBasedExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rat = Ratings(CSVFile(rat_path))
        cls.rat = rat

        cls.gbe = GraphBasedExperiment(rat,
                                       partitioning_technique=KFoldPartitioning(random_state=42, n_splits=2),
                                       algorithm_list=[NXPageRank(alpha=0.8),
                                                       NXPageRank(alpha=0.9)],
                                       items_directory=items_dir,
                                       item_exo_properties={'dbpedia'},
                                       user_exo_properties={'local'},
                                       output_folder="my_experiment")

    def test_rank(self):
        self.gbe.rank(num_cpus=num_cpus)

        # check dirs have been created
        self.assertTrue(os.path.isdir("my_experiment"))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "NXPageRank_1")))
        self.assertTrue(os.path.isdir(os.path.join("my_experiment", "NXPageRank_2")))

        # check user_map and item_map have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "user_map.yml")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "item_map.yml")))

        # check train test splits have been created
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_test_split1.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "KFoldPartitioning_train_split1.csv")))

        # check rank created for nx page rank 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_1", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_1", "rs_rank_split1.csv")))

        # check rank created for nx page rank 2
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_2", "rs_rank_split0.csv")))
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_2", "rs_rank_split1.csv")))

    def test_rank_with_report(self):
        gbe = GraphBasedExperiment(self.rat,
                                   partitioning_technique=KFoldPartitioning(random_state=42, n_splits=2),
                                   algorithm_list=[NXPageRank(alpha=0.8),
                                                   NXPageRank(alpha=0.9)],
                                   items_directory=items_dir,
                                   item_exo_properties={'dbpedia'},
                                   user_exo_properties={'local'},
                                   output_folder="my_experiment",
                                   report=True)

        gbe.rank(num_cpus=num_cpus)

        # check report created for nx page rank 1
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_1", "rs_report.yml")))

        # check report created for nx page rank 2
        self.assertTrue(os.path.isfile(os.path.join("my_experiment", "NXPageRank_2", "rs_report.yml")))

    def test_no_one_predictor(self):

        self.gbe.predict(num_cpus=num_cpus)

        self.assertTrue(os.path.isdir("my_experiment"))

        # since no algorithm is a score prediction one, only train, test split and item and user map will be present
        # and no other thing
        contents_directory = os.listdir("my_experiment")

        contents_directory.remove("KFoldPartitioning_train_split0.csv")
        contents_directory.remove("KFoldPartitioning_test_split0.csv")
        contents_directory.remove("KFoldPartitioning_train_split1.csv")
        contents_directory.remove("KFoldPartitioning_test_split1.csv")
        contents_directory.remove("user_map.yml")
        contents_directory.remove("item_map.yml")

        self.assertTrue(len(contents_directory) == 0)

    def test_predict_raise_error(self):
        with self.assertRaises(NotPredictionAlg):
            self.gbe.predict(num_cpus=num_cpus, skip_alg_error=False)

    def doCleanups(self) -> None:
        # clean folders created
        shutil.rmtree("my_experiment", ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
