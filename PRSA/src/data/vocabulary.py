import os
import random
import logging
import pickle
from typing import List, Any, Dict
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Vocab:

    CLS_TOKEN = '[CLS]'
    START_TOKEN = '[START]'
    END_TOKEN = '[END]'
    UNK_TOKEN = '[UNK]'
    SEP_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    PAD_TOKEN = '[PAD]'
    
    def __init__(self,
                 data: pd.DataFrame=None, 
                 cols_for_vocab: List[str]=None) -> None:
        """
        Initialize the vocabulary with special tokens and the provided columns and data.

        Args:
            - cols_for_vocab (List[str]):
                A list of columns used to create the vocabularies. 
            - data (pd.DataFrame):
                The Pandas DataFrame containing the data for vocabulary creation.
        """
        logger.info('Initializing the vocabulary...')
        # initialize dataset attributes and attribute validation
        self.cols_for_vocab = cols_for_vocab
        self.data = data
        self.token2id = {}
        self.id2token = {}
        self._validate_attributes()
        # initialize the vocabularies with the special tokens
        self._initialize_special_tokens()
        # fill the token2id and id2token mappings and create a lookup table from global_ids to local_ids 
        # (used during training to accelerate the mapping from global to local ids)
        self._create_vocabularies()
        self.lookup_tensor = self._create_lookup_tensor()
        logger.info('Vocabularies successfully created.\n')

    def _validate_attributes(self) -> None:
        """Helper function to validate the class attributes."""
        # checks for the columns and the data
        if self.cols_for_vocab is None or len(self.cols_for_vocab) == 0:
            raise ValueError('"cols_for_vocab" cannot be None or empty.')
        if self.data is None or self.data.empty:
            raise ValueError('Data cannot be None or empty.')
        # checks for the columns in the data
        if not all(col in self.data.columns for col in self.cols_for_vocab):
            missing_cols = [col for col in self.cols_for_vocab if col not in self.data.columns]
            raise ValueError(f"The following columns are missing from the dataframe: {', '.join(missing_cols)}")

    def _add_tokens_to_vocab(self,
                             tokens: List[Any],
                             tag: str) -> None:
        """
        Add tokens from a given field to the vocabularies.

        Args:
            - tokens (List[Any]):
                The list of tokens to add to the vocabulary.
            - tag (str):
                The tag or category for the tokens.
        """
        # fill in the values into token2id and id2token mappings
        self.token2id[tag] = {}
        global_index = len(self.id2token)
        for local_index, token in enumerate(tokens):
            self.token2id[tag][token] = [global_index, local_index]
            self.id2token[global_index] = [token, tag, local_index]
            global_index += 1

    def _initialize_special_tokens(self) -> None:
        """Initialize special tokens, token2id and id2token vocabularies."""
        # store the special tokens
        self.cls_token = Vocab.CLS_TOKEN
        self.start_token = Vocab.START_TOKEN
        self.end_token = Vocab.END_TOKEN
        self.unk_token = Vocab.UNK_TOKEN
        self.sep_token = Vocab.SEP_TOKEN
        self.mask_token = Vocab.MASK_TOKEN
        self.pad_token = Vocab.PAD_TOKEN
        # add the special tokens to the token2id and id2token vocabularies
        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.start_token, self.end_token]
        self.special_tag = 'SPECIAL'
        self._add_tokens_to_vocab(self.special_tokens, self.special_tag)

    def _create_vocabularies(self) -> None:
        """Create token2id and id2token vocabularies based on the provided columns (fields) and data."""
        # for each column extract the unique values and build the mappings
        for col in tqdm(self.cols_for_vocab, desc='Creating vocabularies...'):
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not found in data.")
            unique_values = self.data[col].unique()
            self._add_tokens_to_vocab(unique_values, col)

    def _create_lookup_tensor(self) -> torch.Tensor:
        """
        Create a lookup tensor to map global ids to local ids.
        It constructs a tensor where each index corresponds to a global id and its value is the local id.

        Returns:
            - torch.Tensor: A tensor where each index is a global id and its value is the corresponding local id.
        """
        max_global_id = max(self.id2token.keys())
        lookup_tensor = torch.full((max_global_id + 1,), -100, dtype=torch.long)
        for global_id, value in self.id2token.items():
            lookup_tensor[global_id] = value[2]
        return lookup_tensor

    def get_id(self,
               token: Any,
               tag: str) -> int:
        """
        Retrieve the global index for a token and tag.

        Args:
            - token (Any):
                The token to find.
            - tag (str):
                The tag or category for the token.

        Returns:
            - int:
                The global index of the token.
        """
        # get the vocabulary corresponding to the given field (tag) and from that retrieve the global index of the token
        return self.token2id.get(tag, {}).get(token, self.token2id[self.special_tag][self.unk_token])[0]

    def get_token(self,
                  id: int) -> str:
        """
        Retrieve the token corresponding to an index.

        Args:
            - id (int):
                The index of the token.

        Returns:
            - str:
                The token corresponding to that global index.
        """
        return self.id2token.get(id, [self.unk_token])[0]

    def get_special_tokens(self) -> Dict[str, str]:
        """
        Create the mapping between custom tokens and the standard keys used in BERT-like tokenizers.
        Inspiration was taken by: https://github.com/IBM/TabFormer/blob/main/dataset/vocab.py

        Returns:
            - special_tokens_map (Dict[str, str]):
                The dictionary mapping the custom tokens to the standard keys used in Bert Tokenizer
        """
        special_tokens_map = {}
        # create a mapping between custom special tokens and standard keys in BERT tokenizer
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_tag, token)
            special_tokens_map[key] = token
        return special_tokens_map

    def get_global_ids(self,
                       field_name: str) -> List[int]:
        """
        Get the indeces of the tabular dataset columns.

        Args:
            - field_name (str):
                Name of the field.
        Returns:
            - List[int]:
                List containing the token ids in a given field.
        """
        field_global_ids = [self.token2id[field_name][idx][0] for idx in self.token2id[field_name]]
        return field_global_ids

    def map_global_to_local(self,
                            global_ids: torch.Tensor) -> torch.Tensor:
        """
        Map the global ids to the corresponding field local ids using the lookup tensor. This method is intended to be used during training.

        Args:
            - global_ids (torch.Tensor):
                A tensor containing token global ids.

        Returns:
            - local_ids (torch.Tensor):
                A tensor containing token local ids corresponding to the token global ids.
        """
        lookup_tensor_device = self.lookup_tensor.to(global_ids.device)
        local_ids = lookup_tensor_device[global_ids]
        local_ids.masked_fill_(global_ids == -100, -100)
        return local_ids
    
    def print_vocab_summary(self, 
                            print_special_tokens:bool=True, 
                            print_sample_tokens:bool=True,
                            sample_size:int=5,
                            token_limit_per_column:int=5,
                            print_vocab_size_per_column:bool=True, 
                            print_column_data_types:bool=True, 
                            print_vocab_length:bool=True) -> None:
        """
        Print a summary of the vocabulary based on the provided flags.

        Args:
            print_special_tokens (bool): Whether to print the special tokens. Default to True.
            print_sample_tokens (bool): Whether to print sample tokens from each column. Default to True.
            sample_size (int): Number of columns to sample from. Default to 5.
            token_limit_per_column (int): Number of tokens to show per sampled column. Default to 5.
            print_vocab_size_per_column (bool): Whether to print the size of vocabulary per column. Default to True.
            print_column_data_types (bool): Whether to print the data types of each column. Default to True.
            print_vocab_length (bool): Whether to print the total length of the vocabulary. Default to True.
        """
        if print_special_tokens:
            print(f'Special tokens: {self.special_tokens}\n')
        if print_sample_tokens:
            print("Sampling from the Vocabulary:\n")
            vocab_sample = dict(random.sample(list(self.token2id.items()), sample_size))
            for column, tokens in vocab_sample.items():
                print(f"COLUMN_TAG: {column}")
                print("TOKEN: [GLOBAL IDX, LOCAL IDX]")
                limited_tokens = list(tokens.items())[:token_limit_per_column]
                for token, indices in limited_tokens:
                    print(f"{token}: {indices}")
                print("\n")
        if print_vocab_size_per_column:
            for col in self.cols_for_vocab:
                print(f"Number of tokens in column '{col}': {len(self.token2id[col])}")
            print("\n")
        if print_column_data_types:
            print("Data Types per Column:")
            for col in self.cols_for_vocab:
                print(f"Column '{col}': {self.data[col].dtype}")
            print("\n")
        if print_vocab_length:
            print(f"Total Length of the Vocabulary: {len(self)}")
        
    def save_vocab(self, 
                   vocab_dir: str) -> None:
        """
        Save the vocabularies at the specified path in two formats:  
            - One compatible with BERT tokenizer 
            - One to have easy access to the vocabulary object when loading it for the validation and test set
        Inspiration was taken by: https://github.com/IBM/TabFormer/blob/main/dataset/vocab.py
        
        Args:
            - vocab_dir (str):
                The directory where to save the vocabularies.
        """
        logger.info('Saving vocabularies...')
        if not isinstance(vocab_dir, str):
            raise TypeError(f'"vocab_dir" must be a string. Got {type(vocab_dir)}')
        # create the directory where to store the vocabularies
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        self.vocab_file_for_bert = os.path.join(vocab_dir, f'vocab.nb')
        vocab_object_file_pickle = os.path.join(vocab_dir, f'vocab.pkl')
        # save the vocabularies in a format compatible with BERT tokenizer
        with open(self.vocab_file_for_bert, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)
        # save the vocabulary object to have easy access to the vocabulary when loading it for the validation and test set
        with open(vocab_object_file_pickle, 'wb') as f:
            pickle.dump(self, f)
        logger.info('Vocabularies successfully saved.\n')
            
    @staticmethod
    def load_vocab(vocab_dir: str) -> 'Vocab':
        """
        Load the vocabulary object from the specified path.

        Args:
            - vocab_dir (str):
                The directory where the vocabularies are stored.
        
        Returns:
            - Vocab:
                The vocabulary object.
        """
        logger.info('Loading vocabularies...')
        if not isinstance(vocab_dir, str):
            raise TypeError(f'"vocab_dir" must be a string. Got {type(vocab_dir)}')
        filename = os.path.join(vocab_dir, 'vocab.pkl')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Vocab file not found in {vocab_dir}")
        # load the vocabulary object
        with open(filename, 'rb') as f:
            vocab = pickle.load(f)
        logger.info('Vocab object successfully loaded.\n')
        return vocab

    def __len__(self) -> int:
        """
        Return the length of the vocabulary.

        Returns:
            - int:
                The length of the vocabulary.
        """
        return len(self.id2token)
