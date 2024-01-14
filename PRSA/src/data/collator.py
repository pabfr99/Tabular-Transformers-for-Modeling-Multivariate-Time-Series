from transformers import DataCollatorForLanguageModeling
from typing import List, Dict, Tuple
import torch

class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Custom Data Collator for Tabular Data and Time Series analysis.

    This class inherits from DataCollatorForLanguageModeling from huggingface.
    It is designed to handle tabular and time series where each row consists of multiple columns and each sample consists of multiple rows.
    The collator can be used for Masked Language Modeling tasks.

    Refer to the following link for source code and documentation of DataCollatorForLanguageModelling:
        - https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L607
    """

    def __call__(self,
                 samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates the samples into a batch. It can handle both MLM and regression tasks.
        For MLM tasks, it masks certain tokens in the input ids based on mlm_probability and returns the labels. 
        For regression tasks, it returns the targets.

        Args:
            - samples (List[Tuple[torch.Tensor, torch.Tensor]]):
                List of tuples containing the samples and their targets.

        Returns:
            - Dict[str, Tensor]:
                A dictionary containing input ids, attention masks and targets (MLM labels or regression targets).
        """
        # get the input ids and targets from the samples
        input_ids = [sample[0] for sample in samples]  
        targets = [sample[1] for sample in samples]
        # pad the input ids
        batch = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt") # expected shape [batch, seq_len, ncols]
        if self.mlm:
            # get the shape of the input ids and flatten the samples to mask tokens for MLM
            sz = batch['input_ids'].shape
            input_ids = batch['input_ids'].view(sz[0], -1) # expected shape [batch, seq_len*ncols]
            # mask the tokens with a method from DataCollatorForLanguageModeling
            input_ids, labels = self.torch_mask_tokens(input_ids)
            # reconstruct the initial shape
            batch['input_ids'] = input_ids.view(sz)
            batch['labels'] = labels.view(sz)
        else:
            batch['labels'] = torch.stack(targets)
        return batch
