from transformers import BertConfig
from typing import Optional

class CustomBertConfig(BertConfig):
    """
    Custom config class for a hierarchal Bert Model for Tabular Data and Time Series analysis.

    The BertConfig is the configuration class to store the configuration of a [`BertModel`].

    Refer to the following link for source code and documentation of BertConfig:
        - https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bert/configuration_bert.py#L72
    """

    def __init__(self,
                 ncols: Optional[int]=12,
                 vocab_size: Optional[int]=429,
                 field_hidden_size: Optional[int]=64,
                 hidden_size: Optional[int]=768,
                 num_hidden_layers: Optional[int]=6,
                 num_attention_heads: Optional[int]=8,
                 pad_token_id: Optional[int]=0,
                 mlm_probability: Optional[float]=0.15,
                 **kwargs) -> None:

        """
        Initialize the CustomBertConfig module.

        Args:
            - ncols (int, optional):
                The number of columns in the tabular data. Correspond to the number of 'input_ids' in one row. Default to 12.
            - vocab_size (int, optional):
                Vocabulary size of the model. Defines the number of different tokens that can be represented by the `inputs_ids`. Default to 429.
            - field_hidden_size (int, optional):
                 Hidden size for field embeddings.. Default to 64.
            - hidden_size (int, optional):
                Dimensionality of the encoder layers and the pooler layer.
                Corresponds to the dimensionality of the row embedding. Default to 768.
            - num_hidden_layers (int, optional):
                Number of hidden layers in the Transformer encoder. Default to 6.
            - num_attention_heads (int, optional):
                Number of attention heads for each attention layer in the Transformer encoder. Default to 8.
            - pad_token_int (int, optional):
                Index used for padding. Default to 0.
            - mlm_probability (float, optional):    
                Ratio of tokens to mask for masked language modeling. Default to 0.15.
        """

        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.mlm_probability = mlm_probability