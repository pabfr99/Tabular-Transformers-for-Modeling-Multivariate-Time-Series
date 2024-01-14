import torch
from torch import nn
import torch.nn.init as init
from transformers import BertModel, BertConfig, PreTrainedModel
from transformers.activations import ACT2FN
from src.data.vocabulary import Vocab
from src.models.config import CustomBertConfig

class TabRowEmbeddings(nn.Module):
    """
    Custom embedding class for tabular row data.

    This custom class is designed handle embeddings for tabular data where each row consists of multiple columns.
    Each individual token is mapped to an embedding.
    The sequence is then passed to a transformer encoder to capture relationships between columns.
    A final linear projection transform the embeddings to the desired hidden size.
    """
    def __init__(self,
                 config: CustomBertConfig) -> None:
        """
        Initializes the TabRowEmbeddings class.

        Args:
        - config (CustomBertConfig):
            CustomBertConfig object with attributes:
            - vocab_size: Vocabulary size of the model.
            - field_hidden_size: Hidden size for field embeddings.
            - ncols: The number of columns in the tabular data.
            - hidden_size: Hidden size for the output row embeddings.
            - pad_token_id (optional): Index used for padding.
        """
        super().__init__()

        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0))

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.field_hidden_size,
                                                   nhead=8,
                                                   dim_feedforward=config.field_hidden_size,
                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=6)

        self.linear = nn.Linear(in_features=config.field_hidden_size*config.ncols,
                                out_features=config.hidden_size)
        self._init_model_weights()

    def forward(self,
                input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward step of TabRowEmbeddings.

        Args:
            - input_ids:
                Tensor of shape [batch_size, seq_len, ncols] containing the token ids.

        Returns:
            - input_embeds:
                Tensor of shape [batch_size, seq_len, hidden_size] containing the output row embeddings.
        """

        inputs_embeds = self.word_embeddings(input_ids) #[batch_size, seq_len, ncols, field_hidden_size]
        embeds_shape = inputs_embeds.shape
        # reshape the embeddings
        inputs_embeds = inputs_embeds.view(embeds_shape[0]*embeds_shape[1], embeds_shape[2], -1)  #[batch_size*seq_len, ncols, field_hidden_size]
        # passing through the transformer encoder
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        # reshape the embeddings to have a single row embedding
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0], embeds_shape[1], -1)  # [batch_size, seq_len, ncols*field_hidden_size]
        # final linear projection to hidden size
        inputs_embeds = self.linear(inputs_embeds) # [batch_size, seq_len, hidden_size]
        return inputs_embeds

    def _init_model_weights(self):
        """
        Initializes the weights of the model.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -1.0, 1.0)
                
        
class HierarchicalBertLM(PreTrainedModel):
    def __init__(self,
                 config: BertConfig,
                 vocab: Vocab,
                 mode: str='mlm') -> None:
        """
        Initializes the HierarchicalBertLM class. It can be used for masked LM and regression tasks.
        
        Args:
            - config (CustomBertConfig):
                CustomBertConfig object with attributes:
                - vocab_size: Vocabulary size of the model.
                - field_hidden_size: Hidden size for field embeddings.
                - ncols: The number of columns in the tabular data.
                - hidden_size: Hidden size for the output row embeddings.
                - pad_token_id (optional): Index used for padding.
                - hidden_act (optional): Activation function used in the feedforward layer.
                - layer_norm_eps (optional): Epsilon value for layer normalization.
                - 
            - vocab (Vocab):
                Vocab object containing the vocabulary of the model.
            - mode (str, 'mlm' or 'regression'): 
                Mode of the model. If 'mlm', the model is trained with masked LM. If 'regression', the model is trained for regression.               
        """
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.mode = mode
        # tabular embeddings
        self.tabular_row_embeddings = TabRowEmbeddings(self.config)
        # bert model for sequence of rows
        self.bert = BertModel(config)
        # MLM-specific layers 
        self.mlm_linear = nn.Linear(in_features=config.field_hidden_size,
                                    out_features=config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, 
                                       eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        if isinstance(config.hidden_act, str):
            self.activation_function = ACT2FN[config.hidden_act]
        else:
            self.activation_function = config.hidden_act
        # CrossEntropyLoss for masked LM
        self.loss_fct_mlm = nn.CrossEntropyLoss()
        # precompute the global ids for each column
        if self.mode=='mlm':
            self.precomputed_global_ids = {key: self.vocab.get_global_ids(key) for key in self.vocab.cols_for_vocab}
        # regression-specific layers    
        self.hidden_layer_reg = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output_layer_reg = nn.Linear(config.hidden_size, 2)
        self.loss_fct_reg = nn.MSELoss()    
        self._init_model_weights()
        
    def compute_masked_lm_loss(self, 
                               sequence_output: torch.Tensor, 
                               masked_lm_labels: torch.Tensor, 
                               outputs: tuple) -> torch.Tensor:
        """
        Computes the masked LM loss.
        
        Args:
            - sequence_output (torch.Tensor):
                Tensor of shape [batch_size, seq_len, hidden_size] containing the output of the BERT model.
            - masked_lm_labels (torch.Tensor):
                Tensor of shape [batch_size, seq_len, ncols] containing the masked LM labels.
            - outputs (torch.Tensor):
                Tuple containing the outputs of the BERT model.

        Returns:
            - total_masked_lm_loss (torch.Tensor):
                Tensor containing the total masked LM loss
        """
        # we must reshape the output to reconstruct field embeddings
        output_shape = sequence_output.shape # [batch_size, seq_len, hidden_size]
        expected_shape = [output_shape[0], output_shape[1]*self.config.ncols, -1] # [batch_size, seq_len*ncols, field_hidden_size]
        sequence_output = sequence_output.view(expected_shape) 
        masked_lm_labels = masked_lm_labels.view(expected_shape[0], -1) # [batch_size, seq_len*ncols]
        # pass the output of BERT through the feedforward layer, output shape [batch_size, seq_len*ncols, hidden_size]
        hidden_state = self.mlm_linear(sequence_output)
        hidden_state = self.activation_function(hidden_state)
        hidden_state = self.layer_norm(hidden_state)
        prediction_scores = self.decoder(hidden_state) # [batch_size, seq_len*ncols, vocab_size]
        outputs = (prediction_scores, ) + outputs[2:]
        total_masked_lm_loss = 0
        seq_len = prediction_scores.size(1)
        # get the field names
        field_names = self.vocab.cols_for_vocab
        # iterate over the field names
        for index, key in enumerate(field_names):
            # get the global ids for the field
            col_ids = list(range(index, seq_len, len(field_names)+1))
            global_ids_field = self.precomputed_global_ids[key]
            # remember that prediction_scores has shape [batch_size, seq_len*ncols, vocab_size], so we need to select the right columns.   
            # we select the prediction scores for the particular field (ex: SO2) and from them we select the scores only corresponding to the global ids of the field 
            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # [batch_size, seq_len, K] where K is the number of unique tokens in the field (the vocab size of the field)
            # selection of the masked LM labels for the field
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            # map the global ids to local ids
            masked_lm_labels_field_local = self.vocab.map_global_to_local(global_ids=masked_lm_labels_field)
            # compute the masked LM loss for the field  
            masked_lm_loss_field = self.loss_fct_mlm(prediction_scores_field.view(-1, len(global_ids_field)),
                                                     masked_lm_labels_field_local.view(-1))
            if not torch.isnan(masked_lm_loss_field):
                total_masked_lm_loss += masked_lm_loss_field
        return total_masked_lm_loss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None) -> dict:
        """
        Forward step of HierarchicalBertLM. Works for both masked LM and regression.
        
        Args:
            - input_ids (torch.Tensor):
                Tensor of shape [batch_size, seq_len, ncols] containing the token ids.
            - attention_mask (torch.Tensor):
                Tensor of shape [batch_size, seq_len, ncols] containing the attention mask.
            - labels (torch.Tensor):
                Tensor of shape [batch_size, seq_len, 2] containing the masked LM labels or the regression targets.
        
        Returns:
            - dict:
                Dictionary containing the loss and the predictions (if present).
        """
        # construct the embeddings of the tabular data, output shape [batch_size, seq_len, hidden_size]
        inputs_embeds = self.tabular_row_embeddings(input_ids)
        # pass the time series of rows through BERT
        outputs = self.bert(inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        if self.mode=='mlm':    
            total_loss = self.compute_masked_lm_loss(sequence_output, 
                                                     labels, 
                                                     outputs)
            return {'loss': total_loss}
        # regression task
        elif self.mode=='regression':
            # pass the output of BERT through the hidden layer
            sequence_output = self.hidden_layer_reg(sequence_output)
            sequence_output = self.activation_function(sequence_output)
            sequence_output = self.dropout(sequence_output)
            regression_output = self.output_layer_reg(sequence_output)
            # reshape the output and the labels
            regression_output = regression_output.view(-1, 2)
            labels_reshaped = labels.view(-1, 2)
            total_loss = self.loss_fct_reg(regression_output, labels_reshaped)
            return {'loss': total_loss,
                    'predictions': regression_output, 
                    'labels': labels_reshaped}
        else:
            raise ValueError("Neither masked_lm_labels nor labels are provided for the forward pass.")            
        
    def freeze_model_except_nlayers(self, 
                                    n: int=2) -> None:
        """
        Freezes all the layers of the encoder model except the last n layers.
        
        Args:
            - n (int):
                Number of layers to keep unfrozen. Default to 2.
        """
        # freeze all the layers
        for param in self.tabular_row_embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
        # unfreeze the last n layers
        for param in self.bert.encoder.layer[-n:].parameters():
            param.requires_grad = True
        # unfreeze the regression layers (if present)
        if self.mode=='regression':
            for param in self.hidden_layer_reg.parameters():
                param.requires_grad = True
            for param in self.output_layer_reg.parameters():
                param.requires_grad = True
            
    def _init_model_weights(self) -> None:
        """Initializes the weights of the model."""
        # initialize weights for MLM layers
        if self.mode == 'mlm':
            init.xavier_uniform_(self.mlm_linear.weight)
            self.mlm_linear.bias.data.zero_()
            init.xavier_uniform_(self.decoder.weight)
            self.decoder.bias.data.zero_()
        # initialize weights for regression layers
        if self.mode == 'regression':
            init.xavier_uniform_(self.hidden_layer_reg.weight)
            self.hidden_layer_reg.bias.data.zero_()
            init.xavier_uniform_(self.output_layer_reg.weight)
            self.output_layer_reg.bias.data.zero_()
