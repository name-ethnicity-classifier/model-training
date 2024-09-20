""" file to train and evaluate the model """

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import wandb
import sklearn.metrics
import hashlib
import shutil

import torch
import torch.utils.data
import torch.nn as nn

from src.model import ConvLSTM as Model
from src.utils import create_dataloader, show_progress, onehot_to_string, init_xavier_weights, device, char_indices_to_string, lr_scheduler, write_json, load_json
from src.test_metrics import validate_accuracy, create_confusion_matrix, recall, precision, f1_score, score_plot, binary_recall
import src.xman as xman


torch.manual_seed(0)
np.random.seed(0)


class TrainSetup:
    def __init__(self, model_config: dict):
        self.model_config = model_config

        # model file and name
        self.model_name = model_config["model-name"]

        self.log_path = f"./logs/{self.model_name}"
        self.output_path = f"./outputs/{self.model_name}"

        self.model_file = f"{self.log_path}/model.pt"

        # dataset parameters
        self.dataset_folder = f"./datasets/preprocessed_datasets/{model_config['dataset-name']}"
        self.dataset_path = f"{self.dataset_folder}/dataset.pickle"
        self.test_size = model_config["test-size"]

        with open(f"{self.dataset_folder}/nationalities.json", "r") as f: 
            self.classes = json.load(f) 
            self.total_classes = len(self.classes)

        # hyperparameters
        self.epochs = model_config["epochs"]
        self.batch_size = model_config["batch-size"]
        self.hidden_size = model_config["hidden-size"]
        self.rnn_layers = model_config["rnn-layers"]
        self.dropout_chance = model_config["dropout-chance"]
        self.embedding_size = model_config["embedding-size"]
        self.augmentation = model_config["augmentation"]

        # unpack learning-rate parameters (idx 0: current lr, idx 1: decay rate, idx 2: decay intervall in iterations)
        self.lr = model_config["lr-schedule"][0]
        self.lr_decay_rate = model_config["lr-schedule"][1]
        self.lr_decay_intervall = model_config["lr-schedule"][2]

        # unpack cnn parameters (idx 0: amount of layers, idx 1: kernel size, idx 2: list of feature map dimensions)
        self.kernel_size = model_config["cnn-parameters"][0]
        self.cnn_out_dim = model_config["cnn-parameters"][1]

        # dataloaders for train, test and validation
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=self.test_size, val_size=self.test_size, \
                                                                               batch_size=self.batch_size, class_amount=self.total_classes, augmentation=self.augmentation)

        # resume training boolean
        self.continue_ = model_config["resume"]

        # initialize xman experiment manager
        self.xmanager = xman.ExperimentManager(log_path=self.log_path, continue_=self.continue_)
        self.xmanager.init(
            optimizer="Adam", 
            loss_function="NLLLoss", 
            learning_rate=self.lr, 
            custom_parameters=model_config
        )

    def _validate(self, model, dataset, confusion_matrix: bool=False, plot_scores: bool=False):
        validation_dataset = dataset

        criterion = nn.NLLLoss()
        losses = []
        total_targets, total_predictions = [], []

        with torch.no_grad():
            model.eval()
            for names, targets, _ in tqdm(validation_dataset, desc="validating", ncols=100):
                names = names.to(device=device)
                targets = targets.to(device=device)

                predictions = model(names)
                loss = criterion(predictions, targets.squeeze())
                losses.append(loss.item())

                for i in range(predictions.size()[0]):
                    target_index = targets[i].item()
                    prediction_index = predictions[i].argmax().item()

                    total_targets.append(target_index)
                    total_predictions.append(prediction_index)

        # calculate loss
        loss = np.mean(losses)

        # calculate accuracy
        # accuracy = 100 * sklearn.metrics.accuracy_score(total_targets, total_predictions)
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=0.4)

        # calculate precision, recall and F1 scores
        # precision_scores = sklearn.metrics.precision_score(total_targets, total_predictions, average=None)
        precision_scores = precision(total_targets, total_predictions, classes=self.total_classes)

        # recall_scores = sklearn.metrics.recall_score(total_targets, total_predictions, average=None)
        recall_scores = recall(total_targets, total_predictions, classes=self.total_classes)

        # f1_scores = sklearn.metrics.f1_score(total_targets, total_predictions, average=None)
        f1_scores = f1_score(precision_scores, recall_scores)
    	
        # create confusion matrix
        if confusion_matrix:
            create_confusion_matrix(total_targets, total_predictions, classes=self.classes, save_path=self.log_path)

            # wandb.log({"conf_mat": wandb.plot.confusion_matrix(
            #     probs=None, y_true=total_targets, preds=total_predictions, class_names=self.classes
            # )})

        if plot_scores:
            score_plot(precision_scores, recall_scores, f1_scores, self.classes, save=self.log_path)

        return loss, accuracy, (precision_scores, recall_scores, f1_scores)

    def train(self):
        wandb_id = str(hashlib.sha256(self.model_name.encode("utf-8")).hexdigest())[:8]
        wandb.init(project=self.model_config["wandb-project"], entity=self.model_config["wandb-entity"], id=wandb_id, resume=self.continue_, config=self.model_config)

        model = Model(
            class_amount=self.total_classes,
            hidden_size=self.hidden_size,
            layers=self.rnn_layers,
            dropout_chance=self.dropout_chance,
            embedding_size=self.embedding_size,
            kernel_size=self.kernel_size,
            cnn_out_dim=self.cnn_out_dim
        ).to(device=device)

        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.model_config["wandb-project"]

        iterations = 0
        for epoch in range(1, (self.epochs + 1)):

            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []

            for names, targets, _ in tqdm(self.train_set, desc="epoch", ncols=100):
                optimizer.zero_grad()

                names = names.to(device=device)
                targets = targets.to(device=device)
                predictions = model.train()(names)

                loss = criterion(predictions, targets.squeeze())
                loss.backward()

                lr_scheduler(optimizer, iterations, decay_rate=self.lr_decay_rate, decay_intervall=self.lr_decay_intervall)
                optimizer.step()

                # log train loss
                epoch_train_loss.append(loss.item())
                
                # log targets and prediction of every iteration to compute the train accuracy later
                validated_predictions = model.eval()(names)
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].item())
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))
                
                iterations += 1

                # decay
                if iterations % self.lr_decay_intervall == 0:
                    # wandb.log({"learning rate": optimizer.param_groups[0]["lr"]})
                    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * self.lr_decay_rate

            # calculate train loss and accuracy of last epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = 100 * sklearn.metrics.accuracy_score(total_train_targets, total_train_predictions)

            # calculate validation loss and accuracy of last epoch
            epoch_val_loss, epoch_val_accuracy, scores = self._validate(model, self.validation_set)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy, f1_score=np.mean(scores[-1]))
            print("\nlr: ", optimizer.param_groups[0]["lr"], "\n")

            # log with wandb
            wandb.log({"validation-accuracy": epoch_val_accuracy, "validation-loss": epoch_val_loss, "train-accuracy": epoch_train_accuracy, "train-loss": epoch_train_loss})

            # log epoch results with xman (uncomment if you have the xman libary installed)
            self.xmanager.log_epoch(model, self.lr, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

        # plot train-history with xman (uncomment if you have the xman libary installed)
        self.xmanager.plot_history(save=True)

    def test(self, print_amount: int=None, plot_confusion_matrix: bool=True, plot_scores: bool=True):
        model = Model(
            class_amount=self.total_classes,
            hidden_size=self.hidden_size,
            layers=self.rnn_layers,
            dropout_chance=0.0,
            embedding_size=self.embedding_size,
            kernel_size=self.kernel_size,
            cnn_out_dim=self.cnn_out_dim
        ).to(device=device)

        model.load_state_dict(torch.load(self.model_file))
        _, accuracy, scores = self._validate(model, self.test_set, confusion_matrix=plot_confusion_matrix, plot_scores=plot_scores)
        
        precisions, recalls, f1_scores = scores
        print("\n\ntest accuracy:", accuracy)
        print("precision of every class:", precisions)
        print("recall of every class:", recalls)
        print("f1-score of every class:", f1_scores)

        self.save_model_configuration(accuracy, [precisions, recalls, f1_scores])

        iterations = 0

        # Print some predictions
        if not print_amount or print_amount == 0:
            return

        for names, targets, non_padded_names in tqdm(self.test_set, desc="epoch", ncols=150):
            names = names.to(device=device)
            targets = targets.to(device=device)

            predictions = model.eval()(names)
            predictions, targets, names = predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), names.cpu().detach().numpy()

            for idx in range(len(names)):
                if len(names) == 1:
                    continue

                names, prediction, target, non_padded_name = names[idx], predictions[idx], targets[idx], non_padded_names[idx]

                # convert to one-hot target
                amount_classes = prediction.shape[0]
                target_empty = np.zeros((amount_classes))
                target_empty[target] = 1
                target = target_empty

                # convert log-softmax to one-hot
                prediction = list(np.exp(prediction))
                certency = np.max(prediction)
                
                prediction = [1 if e == certency else 0 for e in prediction]
                certency = round(certency * 100, 4)

                target_class = list(target).index(1)
                target_class = list(self.classes.keys())[list(self.classes.values()).index(target_class)]
                
                try:
                    # catch, if no value is above the threshold (if used)
                    predicted_class = list(prediction).index(1)
                    predicted_class = list(self.classes.keys())[list(self.classes.values()).index(predicted_class)]
                except:
                    predicted_class = "else"

                name = char_indices_to_string(char_indices=non_padded_name)

                print("\n______________\n")
                print("name:", name)

                print("predicted as:", predicted_class, "(" + str(certency) + "%)")
                print("actual target:", target_class)

                iterations += 1

                if iterations == print_amount:
                    return


    def save_model_configuration(self, accuracy: float, scores: list):
        if os.path.exists(self.output_path):
            print("\nError: The directory '{}' does already exist! Reinitializing.\n".format(self.output_path))
            shutil.rmtree(self.output_path)

        os.mkdir(self.output_path)
        write_json(f"{self.output_path}/results.json", 
            {
                "accuracy": accuracy,
                "precision-scores": scores[0],
                "recall-scores": scores[1],
                "f1-scores": scores[2]
            }
        )
        write_json(f"{self.output_path}/config.json", self.model_config)

        shutil.copyfile(f"{self.log_path}/model.pt", f"{self.output_path}/model.pt")
        shutil.copyfile(f"{self.dataset_folder}/nationalities.json", f"{self.output_path}/nationalities.json")

