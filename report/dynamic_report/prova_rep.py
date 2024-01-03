def retrieve_parameter_settings(data, keys):
    current_data = data

    # Iterate through the provided keys to access the desired sub-dictionary
    for key in keys:
        if key in current_data:
            current_data = current_data[key]
        else:
            # if key not found
            return f"Chiave '{key}' non trovata nel dizionario."

    # Check if the last value is None, an empty dictionary, or an empty list
    if current_data in [None, {}, []]:
        return "parameter is null, no setting found."

    # Retrieve the sub-keys and their associated values
    sub_keys_values = [(sub_key, current_data[sub_key]) for sub_key in current_data]

    # Create a string that encloses the found information
    result_string = ", ".join([f"{sub_key}: {sub_value}" for sub_key, sub_value in sub_keys_values])

    return result_string


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
    keys_to_access = ["ContentBasedRS", "algorithm", "LinearPredictor", "only_greater_eq"]
    result = retrieve_parameter_settings(recsys, keys_to_access)

    # Stampare il risultato
    print(result)
