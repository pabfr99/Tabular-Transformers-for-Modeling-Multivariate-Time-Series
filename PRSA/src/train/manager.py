import os
import wandb
from pathlib import Path
from typing import Union
import inspect
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import TrainingArguments, Trainer, BertTokenizerFast
from src.data.vocabulary import Vocab
from src.data.collator import CustomDataCollator
from src.data.dataset import PRSADataset
from src.models.config import CustomBertConfig
from src.models.hierarchical import HierarchicalBertLM

logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self, 
                 model_config_dict: dict,
                 training_config_dict: dict,
                 train_set: PRSADataset,
                 val_set: PRSADataset,
                 test_set: PRSADataset=None,
                 root_dir: str='/content',
                 project_name: str='PRSATabBert',
                 model_name: str='prsa-0',
                 mode: str='mlm',
                 pretrained_model_path: str=None) -> None:
        """
        Initializes the TrainingManager class.

        Args:
            - model_config_dict (dict): configuration dictionary containing model and training parameters to be logged.
            - training_config_dict (dict): configuration dictionary containing training parameters for training_args
            - data_collator (CustomDataCollator): CustomDataCollator object.
            - train_set (PRSADataset): PRSADataset object containing the training data.
            - val_set (PRSADataset): PRSADataset object containing the validation data.
            - test_set (PRSADataset, optional): PRSADataset object containing the test data. Defaults to None.
            - root_dir (str): Root directory of the project.
            - project_name (str): Name of the project.
            - model_name (str): Name of the model.
            - mode (str): Mode of the model. Either 'mlm' or 'regression'.
            - pretrained_model_path (str): Path to the pretrained model checkpoint.
        """
        self.model_config_dict = model_config_dict
        self.model_config =  CustomBertConfig(**self.model_config_dict)
        self.training_config_dict = training_config_dict
        self.vocab = train_set.vocab
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.root_dir = root_dir
        self.project_name = project_name
        self.model_name = model_name
        self.mode = mode
        self.pretrained_model_path = pretrained_model_path
        self._validate_attributes()
        self.setup_directories()
        self.setup_tokenizer()
        self.setup_collator()
        self.setup_wandb()
        self.setup_model()
        self.setup_training()

    def _validate_attributes(self) -> None:
        """Helper function to validate the attributes."""
        validations = [
            (self.model_config_dict, dict, '"model_config_dict" must be a dictionary'),
            (self.training_config_dict, dict, '"training_config_dict" must be a dictionary'),
            (self.vocab, Vocab, '"vocab" must be an instance of Vocab'),
            (self.train_set, PRSADataset, '"train_set" must be an instance of PRSADataset'),
            (self.val_set, PRSADataset, '"val_set" must be an instance of PRSADataset'),
            (self.test_set, (PRSADataset, type(None)), '"test_set" must be an instance of PRSADataset or None'), 
            (self.root_dir, (str, type(None)), '"root_dir" must be a string or None'),
            (self.project_name, str, '"project_name" must be a string'),
            (self.model_name, str, '"model_name" must be a string'),
            (self.mode, str, '"mode" must be a string'),
            (self.pretrained_model_path, (str, type(None)), '"pretrained_model_path" must be a string or None')
        ]
        for var, var_type, err_msg in validations:
            if not isinstance(var, var_type):
                raise TypeError(f'{err_msg}. Got {type(var)} instead.')
        if self.mode not in ['mlm', 'regression']:
            raise ValueError('"mode" must be either "mlm" or "regression"')
        if self.mode == 'regression' and self.pretrained_model_path is None:
            raise ValueError('"pretrained_model_path" is required for "regression" mode')
        if self.mode == 'regression' and not os.path.exists(self.pretrained_model_path):
            raise ValueError(f'"{self.pretrained_model_path}" does not exist')
        valid_params  = set(inspect.signature(TrainingArguments).parameters.keys())
        input_params = set(self.training_config_dict.keys())
        invalid_params = input_params - valid_params
        if invalid_params:
            raise ValueError(f'Invalid parameters in training_config_dict: {invalid_params}')
    
    def setup_directories(self) -> None:
        """Sets up the output and logs directories."""
        try:
            base_dir = Path(self.root_dir) / f'output' / self.mode
            base_dir.mkdir(parents=True, exist_ok=True)
            self.CHECKPOINT_DIR = base_dir / 'checkpoints' / self.model_name 
            self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.LOGS_DIR = base_dir / 'logs' / self.model_name
            self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"An error occurred while setting up directories: {e}")
            raise 
    
    def setup_collator(self) -> None:
        """Sets up the data collator for training."""
        try:
            self.data_collator = CustomDataCollator(tokenizer=self.tokenizer, 
                                                    mlm_probability=self.model_config.mlm_probability, 
                                                    mlm=self.mode=='mlm')
        except Exception as e:
            logger.error(f"An error occurred while setting up the data collator: {e}")
            raise

    def setup_tokenizer(self) -> None:
        """Sets up the tokenizer for training."""
        try:
            self.tokenizer = BertTokenizerFast(vocab_file=self.vocab.vocab_file_for_bert,
                                               do_lower_case=False,
                                               **self.vocab.get_special_tokens())
        except Exception as e:
            logger.error(f"An error occurred while setting up the tokenizer: {e}")
            raise

    def setup_model(self) -> None:
        """
        Sets up the model for training. 
        If the mode is 'mlm', the model is initialized from scratch and trained with masked language modeling. 
        If a pretrained_model_path is provided, the model is initialized from the pretrained checkpoint.
        If the mode is 'regression', the model is initialized from a pretrained checkpoint and trained for regression.
        """
        try: 
            if self.pretrained_model_path:
                self.model = HierarchicalBertLM.from_pretrained(self.pretrained_model_path,
                                                                config=self.model_config,
                                                                vocab=self.vocab,
                                                                mode=self.mode)
            else:
                self.model = HierarchicalBertLM(config=self.model_config, 
                                                vocab=self.vocab, 
                                                mode=self.mode)
            if self.mode == 'regression':
                self.model.freeze_model_except_nlayers(n=2)
        except Exception as e:
            logger.error(f"An error occurred while setting up the model: {e}")
            raise

    def setup_wandb(self) -> None:
        """Sets up wandb for logging."""
        try:
            wandb.init(config=self.model_config_dict, 
                       project=self.project_name, 
                       name=self.model_name,
                       group=self.mode,
                       dir=str(self.LOGS_DIR))
            wandb.config.update(self.model_config_dict)
        except Exception as e:
            logger.error(f"An error occurred while setting up wandb: {e}")
            raise

    def setup_training(self) -> None:
        """Sets up the training arguments and trainer.""" 
        try:
            self.training_args = TrainingArguments(output_dir=str(self.CHECKPOINT_DIR),
                                                   logging_dir=str(self.LOGS_DIR),         
                                                   **self.training_config_dict)
            if self.mode == 'regression':
                self.trainer = Trainer(model=self.model,
                                       args=self.training_args,
                                       data_collator=self.data_collator,
                                       train_dataset=self.train_set,
                                       eval_dataset=self.val_set,
                                       compute_metrics = self.compute_metrics)
            else:
                self.trainer = Trainer(model=self.model,
                                       args=self.training_args,
                                       data_collator=self.data_collator,
                                       train_dataset=self.train_set,
                                       eval_dataset=self.val_set)
        except Exception as e:
            logger.error(f"An error occurred while setting up training: {e}")
            raise
            
    def _cleanup(self) -> None:
        """Helper function to terminate wandb process."""
        if wandb.run:
            wandb.finish()
        logger.info('Cleanup completed.')

    def train(self, 
              resume_from_checkpoint: Union[bool, str]=None) -> None:
        """
        Trains the model.

        Args:
            - resume_from_checkpoint (Union[bool, str], optional): 
                If a str, local path to a saved checkpoint as saved by a previous instance of Trainer. 
                If a bool and equals True, load the last checkpoint in args.output_dir as saved by a previous instance of Trainer. 
                If None, training starts from scratch.
        """
        if not wandb.run:
            self.setup_wandb()
        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
        finally:
            self._cleanup()
            
    def compute_metrics(self, out) -> dict:
        """Computes metrics for the model."""
        predictions = out.predictions[0]
        labels = out.predictions[1]
        rmse = mean_squared_error(labels, predictions, squared=False)
        mae = mean_absolute_error(labels, predictions)
        r_squared = r2_score(labels, predictions)
        return {"rmse": rmse, 
                "mae": mae, 
                "r_squared": r_squared}

    def evaluate(self, 
                 test=False):
        """
        Evaluates the model on the validation or test set based on the value of test.
        
        Args:
            - test (bool): If True, evaluate on the test set. Else, evaluate on the validation set.
        """
        if self.mode == 'regression':
            if test:
                if self.test_set is None:
                    raise ValueError('Test set is None. Cannot evaluate.')
                else:
                    predictions = self.trainer.predict(self.test_set)
                    metrics = self.compute_metrics(predictions)
                    return metrics, predictions.predictions[0], predictions.predictions[1]
        out = self.trainer.evaluate()
        return out
