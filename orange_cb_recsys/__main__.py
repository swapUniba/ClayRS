import sys

from orange_cb_recsys.script_handling import script_run

DEFAULT_CONFIG_PATH = "web_GUI/app/configuration_files/config.json"

if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = DEFAULT_CONFIG_PATH

    script_run(config_path)
