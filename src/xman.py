
import json
import os
import matplotlib.pyplot as plt
import torch


class ExperimentManager:
    def __init__(self, experiment_name: str="new-experiment", continue_: bool=False):
        """ init experiment logger

        :param str directory: path to the directory in which to manage create the 'x-manager' folder (contains the experiments)
        :param bool continue_: set to 'true' if the user wants to continue the training after interrupting,
            if set to 'false' all logged training information in 'train-stats.json' will be deleted when training again
        """

        self.experiment_name = experiment_name
        self.continue_ = continue_

        # xman-folder
        self.parent_directory = "../logs"
        # experiment-folder
        self.experiment_directory = f"{self.parent_directory}/{self.experiment_name}"

        # create x-manager folder if first usage
        if not os.path.exists(self.parent_directory):
            os.mkdir(self.parent_directory)

        # number of current experiments in xman-folder
        self.total_experiments = len(os.listdir(self.parent_directory))

        self.file_ = f"{self.experiment_directory}/train_stats.json"
        self.model = f"{self.experiment_directory}/model.pt"

        # if there is no experiment with that given name, create new one
        if not os.path.exists(self.experiment_directory):
            # create experiment folder
            os.mkdir(self.experiment_directory)
            # create training-stats json file
            open(self.file_, "w+").close()
            # create pytorch-model file
            open(self.model, "w+").close()


    def init(self, optimizer: str="", loss_function: str="", learning_rate: float=0, custom_parameters: dict={}) -> None:
        """ (re-) initializes train-stats json file

        :param str optimizer: name of optimizer (can be chosen by user)
        :param str loss_function: name of loss-function (can be chosen by user)
        :param float learning_rate: (initial) learning-rate
        :param dict custom_parameters: a dictionary of custom hyperparameters (ie. {"learning-rate-decay": 0.9, "residual": True})
            which gets concatinated with the standart hyperparameter-dictionary
        """

        try:
            if repr(type(optimizer)).split(".")[1] == "optim":
                optimizer = repr(type(optimizer)).split(".")[3].split("'")[0]
        except:
            pass

        try:
            if repr(type(loss_function)).split(".")[3] == "loss":
                loss_function = repr(type(loss_function)).split(".")[4].split("'")[0]
        except:
            pass

        # if new training starts initialize training-stats json object
        if not self.continue_:
            entry = {
                "hyperparameters": {
                    "learning-rate": [learning_rate]
                },
                "results": {
                    "train-accuracy": [],
                    "train-loss": [],
                    "validation-accuracy": [],
                    "validation-loss": [],
                }
            }

            if len(custom_parameters) > 0:
                entry["hyperparameters"].update(custom_parameters)
        
            with open(self.file_, "w") as f:
                json.dump(entry, f, indent=4)
        
        else:
            pass
                
    def log_epoch(self, model, lr: float, train_acc: float, train_loss: float, val_acc: float, val_loss: float) -> None:
        """ logs new training results in train-stats json file per epoch

        :param torch-model model: model to load the weights from for checkpoint-saving
        :param float lr: learing-rate (to log if it changes over time, ie. learning-rate decay or scheduler)
        :param float train_acc: train-accuracy of epoch
        :param float train_loss: train-loss of epoch
        :param float val_acc: validation-accuracy of epoch
        :param float val_loss: validation-loss of epoch
        """

        # save model-checkpoint
        torch.save(model.state_dict(), self.model)

        with open(self.file_, "r") as f:
            previous_entry = json.load(f)
        
        current_epoch = len(previous_entry["results"]["train-accuracy"])

        # log changing learning-rate (ie. learning-rate decay or scheduler) with current epoch
        # ie. [0.001, (0.005, 5), (0.0025, 10)] -> indicates learning-rate halving every 5 epochs
        current_lr = lr
        previous_lr = previous_entry["hyperparameters"]["learning-rate"][-1]
        try:
            previous_lr = previous_lr[0]
        except:
            pass
        if current_lr != previous_lr:
            previous_entry["hyperparameters"]["learning-rate"].append((current_lr, current_epoch))

        # log results of current epoch
        previous_entry["results"]["train-accuracy"].append(train_acc)
        previous_entry["results"]["train-loss"].append(train_loss)
        previous_entry["results"]["validation-accuracy"].append(val_acc)
        previous_entry["results"]["validation-loss"].append(val_loss)

        with open(self.file_, "w") as f:
            json.dump(previous_entry, f, indent=4)

    def plot_history(self, save: bool=False) -> None:
        """ plots training histroy

        :param bool save: check to save the plot
        """

        with open(self.file_, "r") as f:
            entry = json.load(f)

        train_acc, train_loss = entry["results"]["train-accuracy"], entry["results"]["train-loss"]
        val_acc, val_loss = entry["results"]["validation-accuracy"], entry["results"]["validation-loss"]
        x_range = list(range(len(train_acc)))

        plt.style.use("bmh")

        fig, axs = plt.subplots(2)
        axs[0].plot(x_range, train_acc, c="b", label="train-acc")
        axs[0].plot(x_range, val_acc, c="r", label="val-acc")
        axs[0].set_title("train- /val-acc")

        axs[1].plot(x_range, train_loss, c="b", label="train-loss")
        axs[1].plot(x_range, val_loss, c="r", label="val-loss")
        axs[1].set_title("train- /val-loss")

        axs[0].legend()
        axs[1].legend()

        if save:
            plt.savefig(f"{self.experiment_directory}/history.png")

        plt.show()
