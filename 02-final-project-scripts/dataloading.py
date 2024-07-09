import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Constants
# ##################################################################################################

TRAINING_DATA_PATH = '../data/sentence-relations/train.csv'
TEST_DATA_PATH     = '../data/sentence-relations/test.csv'

# Pandas Dataframe Loading
# ##################################################################################################

def load_sentence_dataframe(use_training_data=True, only_english=False):

    data_path = TRAINING_DATA_PATH if use_training_data else TEST_DATA_PATH
    raw_data  = pd.read_csv(data_path, index_col='id')

    # Remove chinese
    raw_data = raw_data[raw_data['lang_abv'] != 'zh']
    raw_data = raw_data[raw_data['lang_abv'] != 'th']

    # Only take english
    if only_english:
        raw_data = raw_data[raw_data['lang_abv'] == 'en']

    # Perform a train-test split, when the non submission data is loaded
    if use_training_data:
        training_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
        return training_data, test_data
    else:
        return raw_data


# PyTorch Dataset
# ##################################################################################################

class SentenceRelationTransformerDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data               = data
        self.sentence_relations = []
        self.labels             = []
        self.tokenizer          = tokenizer

        self.perform_preprocessing()

    def __len__(self):
        return len(self.data)

    def get_max_input_length(self):

        max_length = 0

        for sentence_pair in self.sentence_relations:
            max_length = max(max_length, len(sentence_pair[0]), len(sentence_pair[1]))

        return max_length

    def perform_preprocessing(self):
        self.sentence_relations = []
        self.labels             = []

        for i in range(len(self.data)):
            premise    = self.data.iloc[i]['premise']
            hypothesis = self.data.iloc[i]['hypothesis']

            input_string = f"Premise: {premise} | Hypothesis: {hypothesis}"

            input_tokenized = self.tokenizer(input_string, padding='max_length', max_length=512, truncation=True)

            label = self.data.iloc[i]['label']

            self.labels.append(label)
            self.sentence_relations.append(input_tokenized)

    def __getitem__(self, idx):
        temp_dict = self.sentence_relations[idx]
        temp_dict['label'] = self.labels[idx]

        return temp_dict