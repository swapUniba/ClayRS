import re

def retrieve_parameter_settings(data, keys):
    current_data = data

    # Iterate through the provided keys to access the desired sub-dictionary
    for key in keys:
        if key in current_data:
            current_data = current_data[key]
        else:
            # if key not found
            return f"Key '{key}' not found in the dictionary."

    # Check if the last value is None, an empty dictionary, or an empty list
    if current_data in [None, {}, []]:
        return "parameter is null, no setting found."

    # Retrieve the sub-keys and their associated values
    sub_keys_values = [(sub_key, current_data[sub_key]) for sub_key in current_data]

    # Create a string that encloses the found information
    result_string = ", ".join([f"{sub_key}: {sub_value}" for sub_key, sub_value in sub_keys_values])

    return result_string


def process_string(input_string):
    # Separatore per le chiavi primarie
    separator = "\\item "

    # Sostituisci gli underscore con "\\_"
    processed_string = input_string.replace('_', r'\_')

    # Cerca tutte le occorrenze di chiavi e i loro valori nel formato chiave: valore
    matches = re.finditer(r'(\w+):\s*([^,}]+)', processed_string)

    # Itera su tutte le corrispondenze
    for match in matches:
        key = match.group(1)
        value = match.group(2)

        # Aggiungi il separatore e la chiave con il testo successivo al risultato
        processed_string = processed_string.replace(match.group(0), f'{separator}{key} {value}')

    return processed_string


def get_subkey_for_recsys_format(dictionary: dict, *keys_in_order: str) -> list:
    subkeys_at_path = []

    def explore_dictionary(current_dict, current_keys):
        if len(current_keys) == 0 or current_dict is None:
            return

        current_key = current_keys[0]

        if isinstance(current_dict, dict) and current_key in current_dict:
            next_dict = current_dict[current_key]

            if len(current_keys) == 1:
                if isinstance(next_dict, dict):
                    subkeys_at_path.extend([f" {key}" for key in next_dict.keys()])
                else:
                    subkeys_at_path.append(f"{current_key}")
            else:
                explore_dictionary(next_dict, current_keys[1:])

    explore_dictionary(dictionary, keys_in_order)
    return subkeys_at_path

# RUN script
if __name__ == "__main__":
    # Esempio di utilizzo con il dizionario 'recsys' fornito
    recsys = {
        "ContentBasedRS": {
            "algorithm": {
                "LinearPredictor": {
                    "item_field": {
                        "plot": ["tfidf_sk", "kf_renk", "deltaX"],
                        "gener": {
                            "a": 5,
                            "b": 7,
                            "c": {
                                "a": 9,
                                "b": 10
                            }
                        }
                    },
                    "regressor": "SkLinearRegression",
                    "only_greater_eq": None,
                    "embedding_combiner": "Centroid"
                }
            }
        }
    }

    # Esempio di chiamata della funzione
    keys_to_access = ["ContentBasedRS", "algorithm", "LinearPredictor"]
    result = get_subkey_for_recsys_format(recsys, "ContentBasedRS", "algorithm", "LinearPredictor")

    # Stampare il risultato
    print(result)
    """
    input_string = "network: <class 'clayrs.recsys.network_based_algorithm.amar.amar_network.AmarNetworkMerge'>, item_fields: [{'plot': ['tfidf_sk']}, {'plot': ['tfidf_sk']}], user_fields: [{}, {}], batch_size: 512, epochs: 5, threshold: 4, additional_opt_parameters: {'batch_size': 512}, train_loss: <function binary_cross_entropy at 0x00000234EC9A3760>, optimizer_class: <class 'torch.optim.adam.Adam'>, device: cuda:0, embedding_combiner: {'Centroid': {}}, seed: None, additional_dl_parameters: {}"

       processed_string = process_string(input_string)
    """