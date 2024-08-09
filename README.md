# Training a name-ethnicity model

> [!NOTE]  
> Work in progress

This repository contains the pipeline for training name-to-ethnicity models as we used it for our research and on [www.name-to-ethnicity.com](https://www.name-to-ethnicity.com). We ported and cleaned the relevant files from the [nec-experiments](https://github.com/name-ethnicity-classifier/nec-experiments) repository, which was used for experimentation and finding the optimal model.

If you want to simply classify names use the [name-ethnicity-classifier](https://github.com/name-ethnicity-classifier/name-ethnicity-classifier) repository.

> [!NOTE]
> This repository does not contain a dataset. The dataset we used for our research and to train models is private, but we may be able to share it upon request.
> If you have your own dataset, you will find instructions below how to prepare it in order to use it with this repository.

## Setup
1. Clone and enter this repository:
```
git clone ...
cd ...
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Importing a raw dataset
Inside the [./datasets/](./datasets/) folder there is expected to be a file called ``raw_dataset.pickle``, which contains the raw dataset. We didn't include our own dataset as it is private, but here is how you could import your own:

First gather your names at a resource of your choice (CompaniesHouse, Olympic Athletes, etc.) and put them into a JSON file. It's structure should have nationality-names as keys which each have a list of names as a value, for example like this:
```json
{
    "german": ["max mustermann", "frederike maier", "Jospeh Söder"],
    "american": ["joe smith", "Nancy Portman", "meryl moore"],
}
```
Of course you should have many more nationalities and many more names per nationality. Don't worry about special characters or case sensitivity as this will be taken care of automatically.

Next, serialize your JSON file into a ``.pickle`` file, like this:
```python
with open("/path/to/datasets/raw_dataset.pickle", "wb+") as f:
    pickle.dump(your_json_dataset, f, pickle.HIGHEST_PROTOCOL)
```

## Training a model

### 1. Preprocess dataset
TODO

### 2. Training model