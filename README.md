# Training a name-to-ethnicity model

This repository contains the pipeline for training name-to-ethnicity models as we used it for our research and on [www.name-to-ethnicity.com](https://www.name-to-ethnicity.com). We ported and cleaned the relevant files from the [nec-experiments](https://github.com/name-ethnicity-classifier/nec-experiments) repository, which was used for experimentation and finding the optimal model.

If you want to simply classify names use the [name-ethnicity-classifier](https://github.com/name-ethnicity-classifier/name-ethnicity-classifier) repository.

> [!NOTE]
> This repository does not contain a dataset. The dataset we used for our research and to train models is private, but we may be able to share it upon request.
> If you have your own dataset, you will find instructions below how to prepare it in order to use it with this repository.

## ⚙️ Setup
1. Clone and enter this repository:
```
git clone https://github.com/name-ethnicity-classifier/model-training
cd ./model-training/
```

2. Install dependencies:
Make sure you have Python >=3.9 installed and to create a venv/conda environment.
```
pip install -r requirements.txt
```

## 📥 Importing a raw dataset
Inside the [./datasets/](./datasets/) folder there is expected to be a file called ``raw_dataset.pickle``, which contains the raw dataset. We didn't include our own dataset as it is private, but you can reach out via email so that we might share it.

#### Here is how you would import your own dataset:

1. Gather your names at a resource of your choice ([CompaniesHouse](https://www.gov.uk/government/organisations/companies-house), [Olympic Athletes](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results), etc.) and put them into a JSON file. Its structure should have nationality-names as keys which each have a list of names as a value, for example like this:
   ```json
   {
       "german": ["max mustermann", "frederike maier", "Jospeh Söder"],
       "american": ["joe smith", "Nancy Portman", "meryl moore"],
       "indian": ["Priya Srinivasan", "Hakam Ali Banerjee", "Prasun Khan"]
   }
   ```
   Of course you should have many more nationalities and many more names per nationality. 

2. Serialize your JSON file into a ``.pickle`` file, for example like this:
   ```python
   with open("/path/to/datasets/raw_dataset.pickle", "wb+") as f:
       pickle.dump(your_json_dataset, f, pickle.HIGHEST_PROTOCOL)
   ```

3. Update ``./datasets/nationalities.json`` so that it contains a list with all the nationalities that appear in your dataset. You can also map nationalities to bigger groups inside the ``nationality_groups`` entry. If you use our dataset you don't have to change anything here.

If you finally have a ``./datasets/raw_dataset.pickle`` file you are ready to train a model!

## 🦾 Training a model

### 1. Preprocess dataset
First we need to create a preprocessed dataset from the ``./datasets/raw_dataset.pickle`` we added previously by choosing the exact nationalities (or groups) we want. 

For that use ``./create_dataset.py``:

```bash
# Create nationality dataset
python ./create_dataset.py --name <choose dataset name> --classes german greek zimbabwean
```

```bash
# Create nationality group dataset
python ./create_dataset.py --name <choose dataset name> --classes african eastAsian --is_group_level
```
You need to choose your nationalities or groups from the ``./datasets/nationalities.json`` file. You can additionally specify the class ``else`` which will contain random names from all non-specified classes.

When your dataset has been created there will be a new folder ``./datasets/preprocessed_datasetas/<dataset-name>/``.

### 2. Training model
For training use the ``./train_model.py`` file.

To start the training simply run:
```bash
python ./train_model.py --model_name <choose model name> --dataset_name <previously created dataset>
```

You can change all hyperparameters by editing ``./train_model.py`` directly.

> [!TIP]
> If you don't see any improvement on the validation accuracy anymore simply you can simply interrupt the training (CTRL+C). Then to test/validate it comment out ``train_setup.train()`` and run your script again.

You will find training logs inside ``./logs/<model-name>/`` and the final trained model will be inside ``./outputs/<model-name>/``.
To use your model now, please port it to the [name-ethnicity-classifier](https://github.com/name-ethnicity-classifier/name-ethnicity-classifier) repository.
