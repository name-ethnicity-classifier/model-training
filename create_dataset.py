import pickle
import random
import os
import json
import string


random_seed = 123
random.seed(random_seed)


RAW_DATASET_PATH = "./datasets/raw_dataset.pickle"
NATIONALITY_DATA_PATH = "./datasets/nationalities.json"


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


def load_dataset() -> tuple[list, dict]:
    with open(RAW_DATASET_PATH, "rb") as o:
        raw_dataset = pickle.load(o)

    with open(NATIONALITY_DATA_PATH, "r") as f:
        all_nationalities = json.load(f)

    return raw_dataset, all_nationalities


def preprocess_nationalities(dataset_name: str, nationalities: list):
    # Load raw dataset and nationality lists
    entire_dataset, nationality_data = load_dataset()
    all_nationalities = nationality_data["nationalities"]

    # Set minimum names per country
    minimum_per_country = 1

    # Create mapping for letters to indices
    abc_dict = {char: idx for idx, char in enumerate(string.ascii_lowercase + " -")}

    # Filter countries with fewer names than the minimum
    amount_names_country = {key: len(names) for key, names in entire_dataset.items() if len(names) > minimum_per_country}
    entire_dataset = {key: entire_dataset[key] for key in amount_names_country}

    # Initialize chosen nationalities and handle the "else" category
    chosen_nationalities_dict = {nat: [nat] for nat in nationalities if nat != "else"}
    available_nationalities = list(set(all_nationalities) - set(nationalities))

    if "else" in nationalities:
        chosen_nationalities_dict["else"] = available_nationalities

    # Distribute names equally across the chosen countries
    max_per_cluster_dict = max_per_cluster(chosen_nationalities_dict, amount_names_country)
    matrix_name_dict = {}
    nationality_to_number_dict = {}
    number = 0

    for country, names in entire_dataset.items():
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


def preprocess_groups(dataset_name: str, groups: list):
    # Load raw dataset and nationality lists
    entire_dataset, nationality_data = load_dataset()
    all_nationalities, nationality_group_table = nationality_data["nationalities"], nationality_data["nationality_groups"]

    # Create mapping for letters to indices
    abc_dict = {char: idx for idx, char in enumerate(string.ascii_lowercase + " -")}
    
    # Filter and prepare the nationalities based on the provided groups
    nationalities = [nation for group in groups for nation in nationality_group_table.get(group, [])]

    # Include the "else" group if specified
    if "else" in groups:
        else_group = list(set(all_nationalities) - set(nationalities))
        nationalities.append("else")
    else:
        else_group = []

    group_names = [[] for _ in range(len(groups))]

    for country, names in entire_dataset.items():
        group = handle_clusters(country, nationality_group_table)
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


def create_dataset(dataset_name: str, classes: list[str], group_level: bool = False):
    if group_level:
        print(f"Creating dataset using the following nationality groups: {classes}.")
        preprocess_groups(
            dataset_name=dataset_name,
            groups=classes,
        )
    else:
        print(f"Creating dataset using the following nationalities: {classes}.")
        preprocess_nationalities(
            dataset_name=dataset_name,
            nationalities=classes,
        )

    print(f"Saved dataset at ./datasets/preprocessed_datasets/{dataset_name}/")


if __name__ == "__main__":
    import argparse

    # Default parameters, change these if you don't want to use the CLI flags
    dataset_name = "german_spanish_else"
    classes = ["german", "spanish", "else"]
    group_level = False

    # Example for using nationality groups instead (group_level must be true)
    # dataset_name = "african_european_eastasian"
    # classes = ["african", "european", "eastAsian"]
    # group_level = True

    parser = argparse.ArgumentParser(description="Create a preprocessed dataset from the raw dataset by selecting nationalities or groups.")
    
    parser.add_argument("--name", type=str, default=dataset_name, help="Name of the output dataset.")
    parser.add_argument("--classes", nargs="+", default=classes, type=str, help="List of nationalities or groups to include in the dataset.")
    parser.add_argument("--is_group_level", action="store_true", default=group_level, help="Specify this flag if you want to use nationality groups.")

    args = parser.parse_args()

    create_dataset(
        dataset_name=args.name,
        classes=args.classes,
        group_level=args.is_group_level,
    )
