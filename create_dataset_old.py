import pickle
import random
import os
import json
import string


random_seed = 123
random.seed(random_seed)

ALL_NATIONALITIES = [
    "british", "indian", "american", "german", "polish", "pakistani", "italian", "romanian", "french", "chinese", "irish", "japanese", "spanish", "filipino", "dutch", "nigerian", "south korean", "taiwanese", "hong konger","korean", "swiss", "danish", "austrian","belgian", "luxembourger", "portugese",
    "norwegian", "swedish", "finnish", "icelandic", "denmark", "lithuanian", "estonian", "latvian", "hungarian", "bulgarian", "czech", "albanian", "slovak", "slovenian", "algerian", "croatian", "serbian", "macedonian", "georgian", "citizen of bosnia and herzegovina", "kosovan", "belarusian",
    "cypriot", "greek", "russian", "ukrainian", "uzbek", "moldovan", "turkmen","kazakh","kyrgyz", "australian",
    "nepalese", "sri lankan", "singaporean","bangladeshi", "malaysian", "fijian", "thai", "indonesian", "burmese", "vietnamese",
    "turkish", "iraqi", "iranian", "israeli", "yemeni", "syrian", "afghan", "palestinian", "kuwaiti", "armenian", "bahraini", "lebanese", "saudi arabian", "azerbaijani", "emirati","omani","qatari","jordanian","maltese",
    "egyptian", "moroccan", "tunisian","libyan", "cameroonian", "ghanian", "ugandan", "nigerien", "kenyan", "gambian", "ivorian", "senegalese", "eritrean", "sierra leonean", "congolese", "somali", "sudanese", "ethiopian","angolan",
    "zimbabwean", "south african", "zambian", "mauritian", "malawian", "tanzanian", "botswanan", "namibian","citizen of seychelles",
    "canadian", "new zealander","mexican", "dominican", "trinidadian", "barbadian", "kittitian", "st lucian","jamaican","british virgin islander", "costa rican","grenadian","panamanian", "cuban",
    "brazilian", "colombian", "argentinian", "peruvian", "venezuelan", "ecuadorean","chilean", "guyanese","bolivian","uruguayan"
]

MATCHING_TABLE = {"nigerian": "african", "south african": "african", "namibian": "african", "zimbabwean": "african", "ghanian": "african", "egyptian": "african", "kenyan": "african", "mauritian": "african", "ugandan": "african", "cameroonian": "african", "zambian": "african", "congolese": "african", "sierra leonean": "african", "sudanese": "african", "tanzanian": "african", "somali": "african", "malawian": "african", "gambian": "african", "british": "angloAmerican", "irish": "angloAmerican", "american": "angloAmerican", "australian": "angloAmerican", "canadian": "angloAmerican", "new zealander": "angloAmerican", "maltese": "angloAmerican", "british virgin islander": "angloAmerican", "chinese": "eastAsian", "malaysian": "eastAsian", "japanese": "eastAsian", "vietnamese": "eastAsian", "thai": "eastAsian", "korean": "eastAsian", "south korean": "eastAsian", "hong konger": "eastAsian", "taiwanese": "eastAsian", "indonesian": "eastAsian", "polish": "european", "romanian": "european", "italian": "european", "french": "european", "german": "european", "bulgarian": "european", "lithuanian": "european", "dutch": "european", "hungarian": "european", "greek": "european", "latvian": "european", "russian": "european", "belgian": "european", "ukrainian": "european", "slovak": "european", "czech": "european", "swiss": "european", "austrian": "european", "cypriot": "european", "albanian": "european", "estonian": "european", "croatian": "european", "slovenian": "european", "belarusian": "european", "serbian": "european", "moldovan": "european", "kosovan": "european", "filipino": "hispanic", "spanish": "hispanic", "portugese": "hispanic", "brazilian": "hispanic", "mexican": "hispanic", "colombian": "hispanic", "venezuelan": "hispanic", "argentine": "hispanic", "israeli": "Jewish", "pakistani": "arabic", "turkish": "arabic", "bangladeshi": "arabic", "iranian": "arabic", "afghan": "arabic", "iraqi": "arabic", "moroccan": "arabic", "syrian": "arabic", "lebanese": "arabic", "saudi arabian": "arabic", "algerian": "arabic", "jordanian": "arabic", "uzbek": "arabic", "libyan": "arabic", "kazakh": "arabic", "azerbaijani": "arabic", "luxembourger": "european", "georgian": "arabic", "kuwaiti": "arabic", "tunisian": "arabic", "sri lankan": "None", "jamaican": "None", "trinidadian": "None", "swedish": "scandinavian", "denmark": "scandinavian" , "danish": "scandinavian", "norwegian": "scandinavian", "finnish": "scandinavian", "icelandic": "scandinavian", "indian": "southAsian", "singaporean": "southAsian", "nepalese": "southAsian"}


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


def preprocess_nationalities(dataset_name: str, nationalities: str, raw_dataset_path: str):
    # load raw dataset
    with open(raw_dataset_path, "rb") as o:
        dict_chosen_names = pickle.load(o)

    # set lower limit of names per country if wanted
    minimum_per_country = 1

    # abc_dict is a dictionary where the letters "a"-"z" and " " and "-" are keys to lists representing these values in the matrix_name_list
    abc_dict = {}
    abc_list = list(string.ascii_lowercase) + [" ", "-"]
    a = list(range(len(abc_list)))
    for i in range(len(abc_list)):
        abc_dict[abc_list[i]] = a[i]

    # get name amount per country
    amount_names_chountry = {}
    dict_chosen_names_2 = dict_chosen_names.copy()
    for key in dict_chosen_names_2:

        if len(dict_chosen_names[key]) <= minimum_per_country:
            dict_chosen_names.pop(key)
        else:
            amount_names_chountry[key] = len(dict_chosen_names[key])
    
    chosen_nationalities_dict = {}
    available_nationalities = ALL_NATIONALITIES.copy()
    for nationality in nationalities:
        if nationality == "else":
            continue

        chosen_nationalities_dict[nationality] = [nationality]
        available_nationalities.pop(available_nationalities.index(nationality))
        
    if "else" in nationalities:
        chosen_nationalities_dict["else"] = available_nationalities

    # gather equally distributed names of all chosen countries
    max_per_cluster_dict = max_per_cluster(chosen_nationalities_dict, amount_names_chountry)
    matrix_name_dict = {}
    nationality_to_number_dict = {}
    number = 0

    for key in dict_chosen_names:
        try:

            max_nat = max_per_cluster_dict[key]
            counter = 0

            list_of_names = dict_chosen_names[key]
            random.shuffle(list_of_names)

            for name in list_of_names:
                name = name.lower()

                # remove "dr", "ms", "mr", "mrs"
                if name.split(" ")[0] == "dr" or name.split(" ")[0] == "mr" or name.split(" ")[0] == "ms" or name.split(" ")[0] == "miss" or name.split(" ")[0] == "mrs":
                    space_idx = name.strip().index(" ")
                    name = name[space_idx:]

                # remove random spaces before name
                if list(name)[0] == " ":
                    name = name[1:]

                if counter <= max_nat:
                    name = name.strip()
                    nationality = key

                    nationality = handle_clusters(nationality, chosen_nationalities_dict)

                    if nationality not in nationality_to_number_dict and nationality != "other":
                        nationality_to_number_dict[nationality] = number
                        number += 1
                        matrix_name_dict[nationality_to_number_dict[nationality]] = [get_matrix_from_name(name, abc_dict)]
                    elif nationality in nationality_to_number_dict and nationality != "other":
                        matrix_name_dict[nationality_to_number_dict[nationality]] += [get_matrix_from_name(name, abc_dict)]       
                    counter += 1
                else:
                    name = name.strip()
                    nationality = key

                    nationality = handle_clusters(nationality, chosen_nationalities_dict)

                    if nationality not in nationality_to_number_dict and nationality != "other":
                        nationality_to_number_dict[nationality]= number
                        number += 1
                        matrix_name_dict[nationality_to_number_dict[nationality]] = [get_matrix_from_name(name, abc_dict)]
                    elif nationality in nationality_to_number_dict and nationality != "other":
                            matrix_name_dict[nationality_to_number_dict[nationality]] += [get_matrix_from_name(name, abc_dict)]
        except:
            pass

    matrix_name_list = []
    nr_of_countries = 0
    list_countries_used = []

    minimum_per_country = min([len(matrix_name_dict[country]) for country in matrix_name_dict])
    for country in matrix_name_dict:

        if len(matrix_name_dict[country]) >= minimum_per_country:
            list_countries_used.append(country)
            nr_of_countries += 1
            names = matrix_name_dict[country]
            random.shuffle(names)
            names = names[:minimum_per_country]

            for name in names:
                matrix_name_list += [[nr_of_countries, name]]

    random.shuffle(matrix_name_list)

    dataset_path = f"datasets/preprocessed_datasets/{dataset_name}"
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with open(f"{dataset_path}/dataset.pickle", "wb+") as o:
        pickle.dump(matrix_name_list, o, pickle.HIGHEST_PROTOCOL)

    names_countries_used = []
    for i, element in enumerate(list_countries_used):
        country_name = list(nationality_to_number_dict.keys())[list(nationality_to_number_dict.values()).index(element)]
        names_countries_used.append(country_name)

    filepath = f"{dataset_path}/nationalities.json"
    with open(filepath, "w+") as f:
        json.dump(names_countries_used, f, indent=4)


def preprocess_groups(dataset_name: str, groups: str, raw_dataset_path: str):
    # load raw dataset
    with open(raw_dataset_path, "rb") as o:
        dict_chosen_names = pickle.load(o)

    # abc_dict is a dictionary where the letters "a"-"z" and " " and "-" are keys to lists representing these values in the matrix_name_list
    abc_dict = {}
    abc_list = list(string.ascii_lowercase) + [" ", "-"]
    a = list(range(len(abc_list)))
    for i in range(len(abc_list)):
        abc_dict[abc_list[i]] = a[i]

    nationalities = []
    for nationality in ALL_NATIONALITIES:
        if nationality in MATCHING_TABLE and MATCHING_TABLE[nationality] in groups:
            nationalities.append(nationality)

    if "else" in groups:
        nationalities.append("else")

    chosen_nationalities = []
    available_nationalities = ALL_NATIONALITIES.copy()
    for nationality in nationalities:
        if nationality == "else":
            continue

        chosen_nationalities.append(nationality)
        available_nationalities.pop(available_nationalities.index(nationality))
    
    if "else" in nationalities:
        chosen_nationalities.append("else")
        else_ = available_nationalities
    else:
        else_ = []
    
    group_names = [[] for _ in range(len(groups))]

    for country in dict_chosen_names:
        if country in MATCHING_TABLE and MATCHING_TABLE[country] in groups:
            group = MATCHING_TABLE[country]
        elif country in else_:
            group = "else"
        else:
            continue

        class_ = groups.index(group)
        for name in dict_chosen_names[country]:
            try:
                name = name.lower()

                # remove "dr", "ms", "mr", "mrs"
                if name.split(" ")[0] == "dr" or name.split(" ")[0] == "mr" or name.split(" ")[0] == "ms" or name.split(" ")[0] == "miss" or name.split(" ")[0] == "mrs":
                    space_idx = name.strip().index(" ")
                    name = name[space_idx:]

                # remove random spaces before name
                if list(name)[0] == " ":
                    name = name[1:]

                name = name.strip()
                int_name = get_matrix_from_name(name, abc_dict)
                group_names[class_].append([class_ + 1, int_name])
            except Exception as e:
                pass

    maximum_names = len(group_names[0])
    for idx, group in enumerate(group_names[1:]):
        if len(group) < maximum_names:
            maximum_names = len(group)

    dataset = []
    for group in group_names:
        names = group.copy()
        random.shuffle(names)
        dataset += names[:maximum_names]
    
    random.shuffle(dataset)

    dataset_path = f"datasets/preprocessed_datasets/{dataset_name}"
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with open(f"{dataset_path}/dataset.pickle", "wb+") as o:
        pickle.dump(dataset, o, pickle.HIGHEST_PROTOCOL)

    filepath = f"{dataset_path}/nationalities.json"
    with open(filepath, "w+") as f:
        json.dump(groups, f, indent=4)


if __name__ == "__main__":
    raw_dataset_path = "datasets/raw_dataset.pickle"

    # Option 1: Create a dataset based on selected nationalities
    """preprocess_nationalities(
        dataset_name="10_nationalities_and_else",
        nationalities=["british", "german", "chinese", "dutch", "japanese", "indian", "spanish", "italian", "russian", "else"],
        raw_dataset_path=raw_dataset_path
    )"""

    # Option 2: Create a dataset based on selected nationality groups
    preprocess_groups(
        dataset_name="afr_eur_sca_asi",
        groups=["african", "european", "scandinavian", "eastAsian"],
        raw_dataset_path=raw_dataset_path
    )