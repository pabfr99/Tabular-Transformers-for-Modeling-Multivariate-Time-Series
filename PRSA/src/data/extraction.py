import logging 
import os 
import pandas as pd
from tqdm import tqdm 
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class DataExtractor:

    def __init__(self,
                 data_root_dir: str,
                 samples_per_file: Optional[int]=None, 
                 train_size: Optional[float]=0.8, 
                 val_size: Optional[float]=0.1) -> None:
        """
        Initialize the DataExtractor module.

        Args:
            - data_root_dir (str):
                The path to the directory containing the data. The directory must contain one or more csv files.
            - samples_per_file (int, optional):
                The number of samples to extract from each file in the data directory. If None, all the samples will be extracted. Defaults to None.
            - train_size (float, optional):
                The percentage of the data to be used for training. Defaults to 0.8.
            - val_size (float, optional):
                The percentage of the data to be used for validation, the remaining data will be used for testing. Defaults to 0.1.
        """
        logger.info('Initializing the DataExtractor...')
        # helper function to validate the initial inputs
        self.time_cols = ['year', 'month', 'day', 'hour']
        self._validate_initial_inputs(data_root_dir,
                                      samples_per_file, 
                                      train_size,
                                      val_size)
        self.data_root_dir = data_root_dir
        self.samples_per_file = samples_per_file
        self.train_size = train_size
        self.val_size = val_size
        # extract the data into a pandas dataframe
        self._extract_data()
        logger.info("DataExtractor successfully initialized.\n")

    def _validate_initial_inputs(self,
                                 data_root_dir: str,
                                 samples_per_file: Optional[int]=None, 
                                 train_size: Optional[float]=0.7,
                                 val_size: Optional[float]=0.15) -> None:
        """Helper function to validate the initial inputs."""
        if not isinstance(data_root_dir, str) or not os.path.isdir(data_root_dir):
            raise ValueError(f'"data_root_dir" must be a valid directory path. Got {data_root_dir}')
        if samples_per_file is not None:
            if not isinstance(samples_per_file, int) or samples_per_file <= 0:
                raise ValueError(f'"samples_per_file" must be a positive integer or None. Got {samples_per_file}')
        if not isinstance(train_size, float) or not 0 <= train_size <= 1:
            raise ValueError(f'"train_size" must be a float between 0 and 1. Got {train_size}')
        if not isinstance(val_size, float) or not 0 <= val_size <= 1:
            raise ValueError(f'"val_size" must be a float between 0 and 1. Got {val_size}')
        if train_size + val_size > 1:
            raise ValueError(f'The sum of "train_size" and "val_size" must be less than or equal to 1.')
    
    def _merge_time_col(self, 
                        data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the 'year', 'month', 'day', 'hour' columns in a unique column named 'TIMESTAMP'.
        The method to_datetime convert in Unix timestamps in nanoseconds.

        Args:
            - data (pd.DataFrame):
                Pandas DataFrame containing the data.

        Returns:
            - data (pd.DataFrame):
                Pandas DataFrame containing the data with a single 'TIMESTAMP' column containing the Unix timestamps.
        """
        data['TIMESTAMP'] = pd.to_datetime(
            dict(year= data['year'],
                 month= data['month'],
                 day= data['day'],
                 hour= data['hour'])).astype(int)
        # use the min max scaler to transform the time-related data
        scaler = MinMaxScaler()
        data['TIMESTAMP'] = scaler.fit_transform(data['TIMESTAMP'].values.reshape(-1, 1))
        # drop the time-related columns
        data.drop(columns=self.time_cols, inplace=True)
        return data

    def _load_from_csv(self, 
                       file_path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a csv file.

        Args:
            - file_path (str):
                The path to the csv file to be read.

        Returns:
            - pd.DataFrame:
                Pandas DataFrame containing the data.
        """
        # error check for the file_path argument
        if not isinstance(file_path, str):
            raise TypeError('file_path must be a string.')
        # logger.info(f'Loading data from {os.path.split(file_path)[-1]}.')

        # get the dataframe from csv file
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, nrows=self.samples_per_file)
            # check if the dataframe is empty
            if data.empty:
                raise ValueError('Data cannot be empty.')
            # check if the dataframe contains all the required columns
            if not all(col in data.columns for col in self.time_cols):
                missing_cols = [col for col in self.time_cols if col not in data.columns]
                raise ValueError(f"The following columns are missing from the dataframe: {', '.join(missing_cols)}")
            return data
        else:
            raise FileNotFoundError(f'The file at the provided path {os.path.split(file_path)[-1]} was not found.\n')

    def _split_data(self, 
                    data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, validation, and test sets based on timestamp.
        
        Args:
            - data (pd.DataFrame):
                Pandas DataFrame containing the data.
            - train_size (float):
                Proportion of the dataset to include in the train split (0 to 1).
            - val_size (float):
                Proportion of the dataset to include in the validation split (0 to 1).
        
        Returns:
            - train_data (pd.DataFrame):
                Pandas DataFrame containing the training data.
            - val_data (pd.DataFrame):
                Pandas DataFrame containing the validation data.
            - test_data (pd.DataFrame):
                Pandas DataFrame containing the test data.
        """
        data = data.sort_values(by='TIMESTAMP')
        # split the data into train, validation, and test sets
        train_end = int(len(data) * self.train_size)
        val_end = train_end + int(len(data) * self.val_size)
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        return train_data, val_data, test_data
	
    def _extract_data(self) -> None:
        """
        Extract the data from the provided directory.
        If multiple files are present, they are merged into a single Pandas DataFrame.
        The data is then split into train, validation, and test sets.
        """
        # files inside the data directory
        all_files = [os.path.join(self.data_root_dir, file) for file in os.listdir(self.data_root_dir) if os.path.isfile(os.path.join(self.data_root_dir, file))]
        # merge all the files into a DataFrame
        train_dataframes = []
        val_dataframes = []
        test_dataframes = []
        # iterate over the files and extract the data
        for file_path in tqdm(all_files, desc='Data Extraction:'):
            data = self._load_from_csv(file_path)
            data = self._merge_time_col(data)
            train_data, val_data, test_data = self._split_data(data)
            train_dataframes.append(train_data)
            val_dataframes.append(val_data)
            test_dataframes.append(test_data)
        # concatenate the dataframes
        self.train_data = pd.concat(train_dataframes, ignore_index=True)
        self.val_data = pd.concat(val_dataframes, ignore_index=True)
        self.test_data = pd.concat(test_dataframes, ignore_index=True)
        logger.info(
            f'Successfully extracted {len(all_files)} DataFrame. '
            f'Train DataFrame has {len(self.train_data)} rows. '
            f'Validation DataFrame has {len(self.val_data)} rows. '
            f'Test DataFrame has {len(self.test_data)} rows.\n')
