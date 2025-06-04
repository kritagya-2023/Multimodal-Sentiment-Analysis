import pickle
import numpy as np
import json

def convert_ndarray_to_list(data):
    """
    Recursively convert numpy arrays in the data structure to lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    else:
        return data

def inspect_dataset(dataset):
    """
    Inspect the structure and content of a given dataset.

    Parameters:
        dataset (dict): The dataset to inspect, organized as nested dictionaries.

    Outputs:
        Prints detailed information about the keys, data structure, and example content of the dataset.
        Returns a new dictionary with selected fields.
    """
    if not isinstance(dataset, dict):
        print("The dataset provided is not a dictionary.")
        return {}

    # Get all top-level keys in the dataset
    dataset_keys = list(dataset.keys())
    print(f"Top-Level Keys in Dataset: {dataset_keys}\n")

    DATASET_DICT = {}
    # Loop through each top-level key (field) in the dataset
    for field in dataset_keys:
        DATASET_DICT[field] = {}

        if not isinstance(dataset[field], dict):
            print(f"Field '{field}' is not a dictionary. Skipping...")
            continue

        field_keys = list(dataset[field].keys())
        n_keys = len(field_keys)
        if n_keys == 0:
            print(f"Field '{field}' has no subkeys.")
            continue

        print(f"Field: {field}")
        print(f"Number of sub-fields in Field: {n_keys}")
        print(f"Example sub-fields: {field_keys}\n")

        for sub_field in field_keys:
            data_point = dataset[field][sub_field]
            print(f"\tSub-field: {sub_field}")
            print(f"\tType of data: {type(data_point)}")
            if isinstance(data_point, list):
                print(f"\tShape of Data: {len(data_point)}\n")
            else:
                # Assuming the data_point has a shape attribute (e.g., a numpy array)
                try:
                    print(f"\tShape of Data: {data_point.shape}\n")
                except AttributeError:
                    print(f"\tShape of Data: Unknown (no 'shape' attribute)\n")
            if sub_field in ['audio', 'vision', 'id', 'text', 'classification_labels', 'regression_labels']:
                DATASET_DICT[field][sub_field] = data_point

        print("-" * 100)
    return DATASET_DICT

def read_pkl_file(file_path):
    """
    Reads a .pkl file and returns the deserialized data.

    Parameters:
        file_path (str): Path to the .pkl file.

    Returns:
        dict: The data contained in the pickle file.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        print(f"Successfully loaded pickle file: {file_path}")
        return data
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return None

def main(file_path, mode=0):
    dataset = read_pkl_file(file_path)
    if dataset is not None:
        extracted_dataset = inspect_dataset(dataset)
        if mode:
            # Convert numpy arrays to lists so that they become JSON serializable
            serializable_dataset = convert_ndarray_to_list(extracted_dataset)
            output_file = 'extracted_data.json'
            with open(output_file, 'w') as json_file:
                json.dump(serializable_dataset, json_file, indent=4)
            print(f"Extracted dataset saved to {output_file}")

            return {}
        else:
            return extracted_dataset
    else:
        print("Failed to load dataset.")

main("./unaligned_50.pkl")