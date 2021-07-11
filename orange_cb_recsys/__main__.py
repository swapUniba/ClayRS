import json
import sys
import yaml

from orange_cb_recsys.script_handling import script_run

DEFAULT_CONFIG_PATH = "web_GUI/app/configuration_files/config.json"

if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.endswith('.yml'):
        extracted_data = yaml.load(open(config_path), Loader=yaml.FullLoader)
    elif config_path.endswith('.json'):
        extracted_data = json.load(open(config_path))
    else:
        raise ValueError("Wrong file extension")

    script_run(extracted_data)
