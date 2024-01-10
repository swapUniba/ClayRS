import exp_report_generator as exrep
import os

# RUN script
if __name__ == "__main__":
    OUTPUT_TEX = "./../output/report.tex"
    OUTPUT_PATH = "output/report.pdf"
    """
    # test with linear predictor
    LIST_YAML_FILES = ["./../data/data_to_test/item_ca_report_nxPageRank.yml",
                       "./../data/data_to_test/rs_report_linearPredictor.yml",
                       "./../data/data_to_test/eva_report_linearPredictor.yml"]
    """
    """
    # test with yml of Index query
    LIST_YAML_FILES = ["./../data/data_to_test/item_ca_report_nxPageRank.yml",
                       "./../data/data_to_test/rs_report_indexQuery.yml",
                       "./../data/data_to_test/eva_report_indexQuery.yml"]
    """
    """
    # test with yml of classifier recommender
    LIST_YAML_FILES = ["./../data/data_to_test/item_ca_report_nxPageRank.yml",
                       "./../data/data_to_test/rs_report_classifierRecommender.yml",
                       "./../data/data_to_test/eva_report_classifierRecommender.yml"]
    """
    """
    # test with yml of amar double source
    LIST_YAML_FILES = ["./../data/data_to_test/item_ca_report_nxPageRank.yml",
                       "./../data/data_to_test/rs_report_amarDoubleSource.yml",
                       "./../data/data_to_test/eva_report_armarDoubleSource.yml"]
    """
    LIST_YAML_FILES = ["./../data/ca_report.yml", "./../data/rs_report.yml", "./../data/eva_report.yml"]

    # LIST_YAML_FILES = ["./../data/ca_test.yml", "./../data/rs_test.yml", "./../data/eva_test.yml"]
    """
    # test with yml of centroid vector
    LIST_YAML_FILES = ["./../data/data_to_test/item_ca_report_nxPageRank.yml",
                       "./../data/data_to_test/rs_report_centroidVector.yml",
                       "./../data/data_to_test/eva_report_centroidVector.yml"]
    """
    TEMPLATE_FILE = "dynamic_fin_rep.tex"

    MngReport = exrep.DynamicReportManager("./dynamic_fin_rep.tex",
                                           LIST_YAML_FILES[0],
                                           LIST_YAML_FILES[1],
                                           LIST_YAML_FILES[2])

    current_path = os.path.abspath(__file__)
    print("Il percorso dello script in esecuzione è:", current_path)
    MngReport.build_template_file_simplex()
    MngReport.load_template_into_enviroment()
    input_yaml = MngReport.merge_yaml_files(LIST_YAML_FILES, "./../data", "final_report_yml.yml")
    MngReport.generate_dynamic_report(input_yaml, OUTPUT_TEX)
    print()

