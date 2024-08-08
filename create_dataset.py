import pickle
import random
import os
import json
import string


random_seed = 123
random.seed(random_seed)


with open("datasets/nationalities.json", "r") as f:
    nationality_data = json.load(f)

ALL_NATIONALITIES = nationality_data["nationalities"]
NATIONALITY_GROUP_TABLE = nationality_data["nationality_groups"]


def get_matrix_from_name(name: str, abc_dict: list):
    matrix = []
    for letter in name:
        matrix.append(abc_dict[letter])
    return matrix

def get_name_from_matrix(matrix: list, abc_list: list):
    name = ""
    for letter in matrix:
        index = letter
        letter = abc_list[index]
        name += letter
    return name

def handle_clusters(nationality: str, dict_clusters: dict):
    for key in dict_clusters:
        if nationality in dict_clusters[key]:
            return key
    return "other"

def max_per_cluster(cluster_dict: dict, amount_names_country: dict):
    max_per_cluster = {}
    for key in cluster_dict:

        smallest = 1e10
        for country in cluster_dict[key]:

            if country in amount_names_country:
                if amount_names_country[country] <= smallest:
                    smallest = amount_names_country[country]

        for country in cluster_dict[key]:
            max_per_cluster[country] = smallest

    return max_per_cluster

def preprocess_nationalities(dataset_name: str, nationalities: list, raw_dataset_path: str):
    # Load raw dataset
    with open(raw_dataset_path, "rb") as o:
        dict_chosen_names = pickle.load(o)

    # Set minimum names per country
    minimum_per_country = 1

    # Create mapping for letters to indices
    abc_dict = {char: idx for idx, char in enumerate(string.ascii_lowercase + " -")}

    # Filter countries with fewer names than the minimum
    amount_names_country = {key: len(names) for key, names in dict_chosen_names.items() if len(names) > minimum_per_country}
    dict_chosen_names = {key: dict_chosen_names[key] for key in amount_names_country}

    # Initialize chosen nationalities and handle the "else" category
    chosen_nationalities_dict = {nat: [nat] for nat in nationalities if nat != "else"}
    available_nationalities = list(set(ALL_NATIONALITIES) - set(nationalities))

    if "else" in nationalities:
        chosen_nationalities_dict["else"] = available_nationalities

    # Distribute names equally across the chosen countries
    max_per_cluster_dict = max_per_cluster(chosen_nationalities_dict, amount_names_country)
    matrix_name_dict = {}
    nationality_to_number_dict = {}
    number = 0

    for country, names in dict_chosen_names.items():
        max_nat = max_per_cluster_dict.get(country, 0)
        random.shuffle(names)
        
        for idx, name in enumerate(names):
            if idx > max_nat:
                break
            
            name = name.lower().strip()
            if name.split(" ")[0] in ["dr", "mr", "ms", "miss", "mrs"]:
                name = " ".join(name.split(" ")[1:])

            nationality = handle_clusters(country, chosen_nationalities_dict)
            if nationality != "other":
                if nationality not in nationality_to_number_dict:
                    nationality_to_number_dict[nationality] = number
                    number += 1
                    matrix_name_dict[nationality_to_number_dict[nationality]] = []
                matrix_name_dict[nationality_to_number_dict[nationality]].append(get_matrix_from_name(name, abc_dict))

    # Create the final dataset with equally distributed names
    matrix_name_list = []
    minimum_per_country = min(len(names) for names in matrix_name_dict.values())
    list_countries_used = []

    for country_idx, names in matrix_name_dict.items():
        if len(names) >= minimum_per_country:
            list_countries_used.append(country_idx)
            random.shuffle(names)
            matrix_name_list.extend([[country_idx + 1, name] for name in names[:minimum_per_country]])

    random.shuffle(matrix_name_list)

    # Save dataset and metadata
    dataset_path = f"datasets/preprocessed_datasets/{dataset_name}"
    os.makedirs(dataset_path, exist_ok=True)

    with open(f"{dataset_path}/dataset.pickle", "wb+") as o:
        pickle.dump(matrix_name_list, o, pickle.HIGHEST_PROTOCOL)

    country_names = [list(nationality_to_number_dict.keys())[list(nationality_to_number_dict.values()).index(idx)] for idx in list_countries_used]
    
    with open(f"{dataset_path}/nationalities.json", "w+") as f:
        json.dump(country_names, f, indent=4)


def preprocess_groups(dataset_name: str, groups: list, raw_dataset_path: str):
    # Load raw dataset
    with open(raw_dataset_path, "rb") as o:
        dict_chosen_names = pickle.load(o)

    # Create mapping for letters to indices
    abc_dict = {char: idx for idx, char in enumerate(string.ascii_lowercase + " -")}
    
    # Filter and prepare the nationalities based on the provided groups
    nationalities = [nation for group in groups for nation in NATIONALITY_GROUP_TABLE.get(group, [])]

    # Include the "else" group if specified
    if "else" in groups:
        else_group = list(set(ALL_NATIONALITIES) - set(nationalities))
        nationalities.append("else")
    else:
        else_group = []

    group_names = [[] for _ in range(len(groups))]

    for country, names in dict_chosen_names.items():
        group = handle_clusters(country, NATIONALITY_GROUP_TABLE)
        if group in groups or country in else_group:
            class_ = groups.index(group) if group in groups else groups.index("else")
            for name in names:
                try:
                    name = name.lower().strip()

                    # Remove titles and extra spaces
                    if name.split(" ")[0] in ["dr", "mr", "ms", "miss", "mrs"]:
                        name = " ".join(name.split(" ")[1:])

                    int_name = get_matrix_from_name(name, abc_dict)
                    group_names[class_].append([class_ + 1, int_name])
                except Exception:
                    continue

    # Determine the maximum number of names per group
    maximum_names = min(len(group) for group in group_names if group)

    dataset = []
    for group in group_names:
        random.shuffle(group)
        dataset += group[:maximum_names]

    random.shuffle(dataset)

    dataset_path = f"datasets/preprocessed_datasets/{dataset_name}"
    os.makedirs(dataset_path, exist_ok=True)

    with open(f"{dataset_path}/dataset.pickle", "wb+") as o:
        pickle.dump(dataset, o, pickle.HIGHEST_PROTOCOL)

    with open(f"{dataset_path}/nationalities.json", "w+") as f:
        json.dump(groups, f, indent=4)


if __name__ == "__main__":
    raw_dataset_path = "datasets/raw_dataset.pickle"

    # Option 1: Create a dataset based on selected nationalities
    """preprocess_nationalities(
        dataset_name="10_nationalities_and_else_2",
        nationalities=["british", "german", "chinese", "dutch", "japanese", "indian", "spanish", "italian", "russian", "else"],
        raw_dataset_path=raw_dataset_path
    )"""

    # Option 2: Create a dataset based on selected nationality groups
    preprocess_groups(
        dataset_name="afr_eur_sca_asi_3",
        groups=["african", "european", "scandinavian", "eastAsian"],
        raw_dataset_path=raw_dataset_path
    )