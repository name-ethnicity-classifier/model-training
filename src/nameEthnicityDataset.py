
import torch
import numpy as np
import string


def char_indices_to_string(char_indices: list=[str]) -> str:
    alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]
    name = ""
    for idx in char_indices:
        if int(idx) == 0:
            pass
        else:
            name += alphabet[int(idx) - 1]
    
    return name


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10, augmentation: float=0.0):
        """ constructor

        :param list dataset: dataset list
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.dataset = dataset

        # print(len(self.dataset), len(self.dataset) / 20)
        self.class_amount = class_amount

        self.augmentation = augmentation
        self.seperat_dataset = self.dataset.copy()
        np.random.shuffle(self.seperat_dataset)

        self.augmentation_list = [0.0 for _ in range(class_amount)]

    def _preprocess_targets(self, int_representation: int, one_hot: bool=True) -> list:
        """ create one-hot encoding of the target or not

        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        int_representation -= 1

        if one_hot:
            one_hot_target = np.zeros((self.class_amount))
            one_hot_target[int_representation] = 1

            return one_hot_target
        else:
            return [int_representation]

    def _name_switch(self, org_name: list, class_: int, chance: float=0.3) -> list:
        """ switches first and last name part of the name with a random name of the same nationality """

        augmentation_choice = np.random.uniform(0.0, 1.0)
        if augmentation_choice <= chance:

            same_nat_name = []
            for idx, sample in enumerate(self.seperat_dataset):
                if class_ == sample[0]:
                    same_nat_name = [e+1 for e in sample[1]]
                
                # some names have only one part for some reason, so don't break the same-nationality search when such appear
                if 27 in same_nat_name:
                    self.seperat_dataset.pop(idx)
                    break
            
            org_prename, org_surname = self._split_name(org_name)
            same_nat_prename, same_nat_surname = self._split_name(same_nat_name)

            flip_case = np.random.choice([0, 1])

            if flip_case == 0:
                #print(char_indices_to_string(org_name), " | ", char_indices_to_string(same_nat_name), " | ", char_indices_to_string(org_prename + [27] + same_nat_surname))
                return org_prename + [27] + same_nat_surname

            elif flip_case == 1:
                #print(char_indices_to_string(org_name), " | ", char_indices_to_string(same_nat_name), " | ", char_indices_to_string(same_nat_prename + [27] + org_surname))
                return same_nat_prename + [27] + org_surname

        else:
            return org_name

    def _split_name(self, int_name: list) -> list:
        try:
            str_index_name = "".join([str(e) + " " for e in int_name])
            str_index_name_split = str_index_name.split("27", 1)

            pre_int_name, sur_int_name = str_index_name_split[0], str_index_name_split[1]
            pre_int_name = [int(e) for e in pre_int_name.split() if e.isdigit()]
            sur_int_name = [int(e) for e in sur_int_name.split() if e.isdigit()]

            return pre_int_name, sur_int_name
            
        except:
            # the case, when the name is only one word (no first-/ sur-name)
            return int_name, int_name

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][1], self.dataset[idx][0]

        int_name = [e + 1 for e in sample]
        target = self._preprocess_targets(target, one_hot=False)

        augmentation_chance = self.augmentation
        if augmentation_chance > 0.0:
            int_name = self._name_switch(int_name, target, chance=augmentation_chance)
        
        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_sample = [e + 1 for e in sample]

        return torch.Tensor(int_name), torch.Tensor(target).type(torch.LongTensor), non_padded_sample

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)


