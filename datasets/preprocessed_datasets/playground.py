import pickle
import string
import json


def inspect_raw():
    with open("datasets/raw_dataset.pickle", "rb") as f:
        dataset = pickle.load(f)
        pass

def get_name_from_matrix(matrix: list):
    abc_list = list(string.ascii_lowercase) + [" ", "-"]

    name = ""
    for letter in matrix:
        index = letter
        letter = abc_list[index]
        name += letter
    return name


def check():

    from tqdm import tqdm

    with open("datasets/preprocessed_datasets/afr_eur_sca_asi/dataset.pickle", "rb") as f:
        dataset1 = pickle.load(f)

    with open("datasets/preprocessed_datasets/afr_eur_sca_asi_3/dataset.pickle", "rb") as f:
        dataset2 = pickle.load(f)

    with open("datasets/preprocessed_datasets/afr_eur_sca_asi/nationalities.json", "r") as f:
        nat_list = json.load(f)

    all_names_1 = []
    all_nationalities_1 = []
    for d in tqdm(dataset1):
        nationality = nat_list[d[0] - 1]
        name = get_name_from_matrix(d[1])

        all_names_1.append(name)
        all_nationalities_1.append(nationality)

    all_names_2 = []
    all_nationalities_2 = []
    for d in tqdm(dataset2):
        nationality = nat_list[d[0] - 1]
        name = get_name_from_matrix(d[1])

        all_names_2.append(name)
        all_nationalities_2.append(nationality)

    
    not_in = 0
    for idx in range(len(all_names_1)):
        name = all_names_1[idx]
        nat = all_nationalities_1[idx]
        if nat == "russian":
            if name not in all_names_2:
                print(False)
            else:
                print(True)

    print(not_in)


def convert_table():
    from collections import defaultdict

    # Original MATCHING_TABLE
    MATCHING_TABLE = {"nigerian": "african", "south african": "african", "namibian": "african", "zimbabwean": "african", "ghanian": "african", "egyptian": "african", "kenyan": "african", "mauritian": "african", "ugandan": "african", "cameroonian": "african", "zambian": "african", "congolese": "african", "sierra leonean": "african", "sudanese": "african", "tanzanian": "african", "somali": "african", "malawian": "african", "gambian": "african", "british": "angloAmerican", "irish": "angloAmerican", "american": "angloAmerican", "australian": "angloAmerican", "canadian": "angloAmerican", "new zealander": "angloAmerican", "maltese": "angloAmerican", "british virgin islander": "angloAmerican", "chinese": "eastAsian", "malaysian": "eastAsian", "japanese": "eastAsian", "vietnamese": "eastAsian", "thai": "eastAsian", "korean": "eastAsian", "south korean": "eastAsian", "hong konger": "eastAsian", "taiwanese": "eastAsian", "indonesian": "eastAsian", "polish": "european", "romanian": "european", "italian": "european", "french": "european", "german": "european", "bulgarian": "european", "lithuanian": "european", "dutch": "european", "hungarian": "european", "greek": "european", "latvian": "european", "russian": "european", "belgian": "european", "ukrainian": "european", "slovak": "european", "czech": "european", "swiss": "european", "austrian": "european", "cypriot": "european", "albanian": "european", "estonian": "european", "croatian": "european", "slovenian": "european", "belarusian": "european", "serbian": "european", "moldovan": "european", "kosovan": "european", "filipino": "hispanic", "spanish": "hispanic", "portugese": "hispanic", "brazilian": "hispanic", "mexican": "hispanic", "colombian": "hispanic", "venezuelan": "hispanic", "argentine": "hispanic", "israeli": "Jewish", "pakistani": "arabic", "turkish": "arabic", "bangladeshi": "arabic", "iranian": "arabic", "afghan": "arabic", "iraqi": "arabic", "moroccan": "arabic", "syrian": "arabic", "lebanese": "arabic", "saudi arabian": "arabic", "algerian": "arabic", "jordanian": "arabic", "uzbek": "arabic", "libyan": "arabic", "kazakh": "arabic", "azerbaijani": "arabic", "luxembourger": "european", "georgian": "arabic", "kuwaiti": "arabic", "tunisian": "arabic", "sri lankan": "None", "jamaican": "None", "trinidadian": "None", "swedish": "scandinavian", "denmark": "scandinavian" , "danish": "scandinavian", "norwegian": "scandinavian", "finnish": "scandinavian", "icelandic": "scandinavian", "indian": "southAsian", "singaporean": "southAsian", "nepalese": "southAsian"}

    # Create a reverse mapping
    GROUP_TO_NATIONALITIES = defaultdict(list)

    for nationality, group in MATCHING_TABLE.items():
        GROUP_TO_NATIONALITIES[group].append(nationality)

    # Convert defaultdict to a regular dictionary
    GROUP_TO_NATIONALITIES = dict(GROUP_TO_NATIONALITIES)

    # Example to print the new structure
    print(json.dumps(GROUP_TO_NATIONALITIES, indent=4))

check()