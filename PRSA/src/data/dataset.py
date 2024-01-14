import os
import json
import logging 
import numpy as np
import pickle as pkl
from tqdm import tqdm
from typing import Optional, List, Tuple
import pandas as pd
from scipy import stats
from src.data.vocabulary import Vocab
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PRSADataset(Dataset):

    COLS_TO_DISCRETIZE = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN', 'TIMESTAMP']
    COLS_FOR_VOCAB = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN', 'TIMESTAMP', 'wd']
    COLS_TO_DROP = ['No']
    TARGET_COLS = ['PM2.5', 'PM10']

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str='train', 
                 vocab_dir: Optional[str]='vocab', 
                 save_dir: Optional[str]='data/processed',
                 cols_to_discretize: Optional[List[str]]=None,
                 cols_for_vocab: Optional[List[str]]=None,
                 cols_to_drop: Optional[List[str]]=None,
                 target_cols: Optional[List[str]]=None,
                 sequence_length: Optional[int]=10,
                 stride: Optional[int]=5) -> None:
        """
        Initialize the PRSADataset module.

        Args:
            - data (pd.DataFrame):
                The DataFrame containing the data.
            - mode (str, optional):
                The mode of the dataset. Can be 'train', 'val' or 'test'. Default to 'train'.
            - 'vocab_dir' (str, optional):   
                When mode is 'train', the vocabulary is saved in this directory.
                When mode is 'val' or 'test', the vocabulary is loaded from this directory. Default to 'vocab'.
            - save_dir (str, optional):
                The directory where to save the class instance. Default to 'data/processed'.
            - cols_to_discretize (List[str], optional):
                List of columns to discretize. If not provided, defaults to class-level constant ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN', 'TIMESTAMP'].
            - cols_for_vocab (List[str], optional):
                List of columns to be used for the vocabulary. If not provided, defaults to class-level constant ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM', 'RAIN', 'TIMESTAMP', 'wd'].
            - cols_to_drop (List[str], optional):
                List of columns to drop from the dataset. If not provided, defaults to class-level constant ['No'].
            - target_cols (List[str], optional):
                List of columns to be used as targets. If not provided, defaults to class-level constant ['PM2.5', 'PM10'].    
            - sequence_length (int, optional):
                The numbers of subsequent row to consider as a sequence. Default to 10.
            - stride (int, optional):
                The step of the sliding window when combining subsequent rows. Default to 5.         
        """
        logger.info('Initializing the PRSADataset...')
        # initialize dataset attributes
        self.data = data.copy()
        self.mode = mode
        self.vocab_dir = vocab_dir
        # initialize the columns to discretize, to drop and the target columns
        self.cols_to_discretize = cols_to_discretize or PRSADataset.COLS_TO_DISCRETIZE.copy()
        self.cols_for_vocab = cols_for_vocab or PRSADataset.COLS_FOR_VOCAB.copy()
        self.cols_to_drop = cols_to_drop or PRSADataset.COLS_TO_DROP.copy()
        self.target_cols = target_cols or PRSADataset.TARGET_COLS.copy()
        # initialize attributes for the sequences
        self.sequence_length = sequence_length
        self.stride = stride
        self.samples, self.targets = [], []
        # attributes validation
        self._validate_attributes()
        self._preprocess_data(mode=self.mode)
        self._tokenize_data()
        self._prepare_samples()
        self.save(save_dir)
        logger.info('PRSADataset successfully initialized.\n')
    
    def _validate_attributes(self) -> None:
        """Helper function to validate the attributes of the class."""
        # handle the validation of attributes using tuple (variable, expected type, error message)
        validations = [
            (self.data, pd.DataFrame, '"data" must be a pandas DataFrame'),
            (self.mode, str, '"mode" must be a string'),
            (self.vocab_dir, str, '"vocab_dir" must be a string'),
            (self.sequence_length, int, '"sequence_length" must be an integer'),
            (self.stride, int, '"stride" must be an integer'),
            (self.cols_to_discretize, (list, type(None)), '"cols_to_discretize" must be a list or None'),
            (self.cols_to_drop, (list, type(None)), '"cols_to_drop" must be a list or None'),
            (self.target_cols, (list, type(None)), '"target_cols" must be a list or None')]
        # check if the mode is valid
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {self.mode}. Must be one of "train", "val" or "test".')
        # iterate over the list of tuples (variable, expected type, error message)
        for var, expected_type, error_msg in validations:
            if not isinstance(var, expected_type):
                raise TypeError(f'{error_msg}. Got {type(var)}')
        # check positivity of samples_per_file, sequence_length and stride
        if self.sequence_length <= 0 or self.stride <= 0:
            raise ValueError(f'"sequence_length" and "stride" must be positive integers.')
        # check that the data is not None or empty
        if self.data is None or self.data.empty:
            raise ValueError('Data cannot be None or empty.')
        # check if the necessary columns are in the DataFrame
        required_cols = set(self.cols_to_discretize + self.target_cols + self.cols_for_vocab + self.target_cols)
        if not all(col in self.data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            raise ValueError(f"The following columns are missing from the dataframe: {', '.join(missing_cols)}")
        
    def _compute_number_bins(self,
                             col_data: pd.Series) -> int:
        """
        Compute the number of bins to discretize a dataset based on its interquartile range (IQR).
        The method uses the Freedman-Diaconis Rule to compute the width of each bin.
        The rule is robust to outliers and is given as 2*IQR/cubic_root(num_observations).
        The number of bins is calculated as (max_value - min_value)/bin_width.

        Args:
            - col_data (pd.Series):
                The data series to be discretized.

        Returns:
            - int:
                The number of bins to be used for discretization.
        """
        IQR = stats.iqr(col_data, rng=(25,75), nan_policy='omit')
        bin_width = 2 * IQR / np.cbrt(len(col_data.unique()))
        range = np.max(col_data) - np.min(col_data)
        n_bins = int(range/bin_width)
        return n_bins

    def _compute_bin_labels(self,
                            col_data: pd.Series,
                            n_bins: int) -> np.ndarray:
        """
        Compute the bin labels based on the number of bins specified.
        The labels serve as the edges for each bin.

        Args:
            - col_data (pd.Series):
                The data series for which the bin labels are to be computed.
            - n_bins (int):
                The number of bins for quantile calculation.

        Returns:
            - np.ndarray:
                The unique bin labels, which are the edges for each bin.
        """
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_labels = np.quantile(col_data, quantiles)
        bin_labels = np.unique(bin_labels)
        return bin_labels

    def _discretize_column(self,
                           col: str, 
                           bin_labels: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Helper function to discretize a single column.
        If bin_labels is not provided, the number of bins and bin labels are computed.
        
        Args:
            - col (str):
                The column to be discretized.
            - bin_labels (np.ndarray, optional):
                The bin labels to be used for discretization. If not provided, they are computed.
        
        Returns:
            - np.ndarray:
                The bin labels used for discretization.
        """
        # different behaviour for the RAIN column to avoid nan values
        if bin_labels is None:
            if col == 'RAIN':
                bin_labels = np.arange(0, 40, 5)
            else:
                n_bins = self._compute_number_bins(self.data[col])
                bin_labels = self._compute_bin_labels(self.data[col], n_bins) 
        # subtract the value with the closest bin label
        # self.data[col] = self.data[col].apply(lambda x: min(bin_labels, key=lambda label: abs(label - x)))
        self.data[col] = self.data[col].apply(lambda x: bin_labels[np.argmin(np.abs(bin_labels - x))])
        return bin_labels
    
    def _discretize_data(self,
                         save_stats: bool=True) -> None:
        "Discretize the data. The columns specified in self.cols_to_discretize are discretized based on the Freedman-Diaconis Rule."

        # discretize the data using the Freedman-Diaconis Rule.
        logger.info('Starting the discretization process...')
        # filling na values by interpolating 
        self.data[self.cols_to_discretize] = self.data[self.cols_to_discretize].interpolate()
        self.data[self.target_cols] = self.data[self.target_cols].interpolate()
        # discretize each column
        if save_stats:
            info = {}
            for col in tqdm(self.cols_to_discretize, desc='Discretizing columns'):
                # discretize the column and save the bin labels
                info[col] = self._discretize_column(col).tolist()
            # save the bin labels in a json file
            with open(os.path.join(self.vocab_dir, 'bin_stats.json'), 'w') as f:
                json.dump(info, f)
        else:
            # load the bin labels from the json file
            with open(os.path.join(self.vocab_dir, 'bin_stats.json'), 'r') as f:
                info = json.load(f)
            for col in tqdm(self.cols_to_discretize, desc='Applying saved discretization'):
                self._discretize_column(col, np.array(info[col]))
        logger.info('Discretization process completed.\n')

    def _create_and_save_vocab(self) -> None:
        """When mode is 'train', create and save the vocabulary in the specified directory."""
        self.vocab = Vocab(data=self.data, 
                           cols_for_vocab=self.cols_for_vocab)
        self.vocab.save_vocab(vocab_dir=self.vocab_dir)

    def _preprocess_data(self, 
                         mode: str) -> None:
        """
        Preprocess the data based on the mode. 
        - If mode is 'train', columns are dropped and the data is discretized. The vocabulary is created and saved.
        - If mode is 'val' or 'test', the vocabulary is loaded and the data is discretized.
        """
        logger.info(f'Preprocessing the {mode} data...')
        # drop the unneeded columns
        if self.cols_to_drop:
            self.data.drop(columns=self.cols_to_drop, inplace=True)
            logger.info(f'Dropped {self.cols_to_drop} columns.')
        if self.mode == 'train':
            # data preprocessing (drop columns and discretize), then create and save the vocabulary
            self._discretize_data(save_stats=True)
            self._create_and_save_vocab()
        else:
            # apply the same preprocessing as in train mode, then load the vocabulary
            self._discretize_data(save_stats=False)
            try:
                self.vocab = Vocab.load_vocab(vocab_dir=self.vocab_dir)
            except FileNotFoundError:
                raise FileNotFoundError(f'Vocabulary not found in {self.vocab_dir}. Please run the script in train mode first.')
        logger.info(f'{mode} data successfully preprocessed.\n')
        
    def _tokenize_data(self) -> None:
        """Map the tokens in "cols_for_vocab" to the corresponding indices."""
        logger.info('Converting data to indices...')
        self.tokenized_data = self.data.copy()
        # apply the get_id function to each element of the dataframe to get the id
        for col in tqdm(self.cols_for_vocab, desc='Tokenizing columns...'):
            self.tokenized_data[col] = self.data[col].apply(lambda x: self.vocab.get_id(x, col))
        logger.info('Tokenization process completed.\n')

    def _prepare_samples(self) -> None:
        """
        Structuring the samples and the targets for Time-Series Analysis.
        A single sample contains seq_len*ncols token ids, representing a sequence of registrations in the tabular data.
        The number of samples obtained in the end depends on the stride and on the number of subsequent rows considered for each sample.
        """
        logger.info('Preparing samples and targets...')
        sep_id = self.vocab.get_id(self.vocab.sep_token, self.vocab.special_tag)
        # get the column indices
        feature_col_indices = [self.tokenized_data.columns.get_loc(c) for c in self.cols_for_vocab]
        target_col_indices = [self.tokenized_data.columns.get_loc(c) for c in self.target_cols]
        # convert the data to numpy for faster operations
        data_numpy = self.tokenized_data.to_numpy()
        # group by station and iterate through groups to prepare the samples
        groups = self.tokenized_data.groupby('station')
        for _, group_indices in tqdm(groups.groups.items(), desc='Preparing Samples'):
            # extract the numpy data from those indices
            station_data = data_numpy[group_indices]
            # get the number of rows of the data
            nrows = len(station_data) - self.sequence_length
            # iterate throught the rows with the specified stride
            for start_id in range(0, nrows, self.stride):
                sample, target = [], []
                slice_end = start_id + self.sequence_length
                # extract one batch of rows, dividing between sample and targets
                sliced_data = station_data[start_id:slice_end]
                sample_values = sliced_data[:, feature_col_indices]
                target_values = sliced_data[:, target_col_indices]
                # add the sep token at the end of the sample
                sample = np.hstack((sample_values, np.full((self.sequence_length, 1), sep_id))).ravel()
                target = target_values.tolist()
                self.samples.append(sample)
                self.targets.append(target)
        logger.info('Samples and targets successfully organized.\n')

    def get_ncols(self) -> int:
        """
        Retrieve the number of columns used for the vocabulary.

        Returns:
            -int:
                number of columns used for the vocabulary.
        """
        return len(self.cols_for_vocab) + 1


    def __len__(self) -> int:
        """
        Retrieve the length of the dataset.

        Returns:
            - int:
                The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self,
                    index: int)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the sample and the target at the specified index.

        Args:
            - index (int):
                The index of the sample.

        Returns:
            - tuple: A tuple containing:
                - 'sample': The tensor containing the sample values.
                - 'target': The tensor containing the target values.
        """
        sample = torch.tensor(self.samples[index].tolist(), dtype=torch.long).reshape(self.sequence_length, -1)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return sample, target
            
    def save(self, 
             data_dir: str) -> None:
        """
        Save the class instance to a file using pickle.

        Args:
        - data_dir (str): The path where to save the class instance.
        """
        logger.info('Saving PRSA Dataset...')
        if not isinstance(data_dir, str):
            raise TypeError(f'"data_dir" must be a string. Got {type(data_dir)}')
        # create the directory where to store the vocabularies
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # save the class instance
        self.prsa_file = os.path.join(data_dir, f'prsa_{self.mode}.pkl')
        with open(self.prsa_file, 'wb') as file:
            pkl.dump(self, file)
        logger.info(f'Class instance successfully saved.\n')
    
    @staticmethod
    def load(data_dir: str,
             mode: str) -> 'PRSADataset':
        """
        Load a class instance from a file using pickle.

        Args:
            - data_dir (str): The path of the directory to load the class instance from.

        Returns:
            - PRSADataset: An instance of the PRSADataset class.
        """ 
        logger.info('Loading PRSA Dataset...')
        if not isinstance(data_dir, str):
            raise TypeError(f'"data_dir" must be a string. Got {type(data_dir)}')
        # check if the mode is valid and if the file exists
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid mode: {mode}. Must be one of "train", "val" or "test".')
        filename = os.path.join(data_dir, f'prsa_{mode}.pkl')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"PRSA file not found in {data_dir}")
        # load the class instance
        with open(filename, 'rb') as file:
            prsa = pkl.load(file)
        logger.info(f'Class instance successfully loaded.\n')
        return prsa
