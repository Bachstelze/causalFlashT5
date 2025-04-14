# distributed under the GNU AGPL v3 license

import torch

"""
from torch import vmap
# map the function with the batch in the forward function
    def forward(self, x):
        y = vmap(causal_max_reduction)(x)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
"""

def causal_mean_reduction(batch):
    """
    vmap can't call a function with many parameters,
    so this function routes it to the algorithm.
    """
    return causal_reduction(batch, "mean")

def causal_max_reduction(batch):
    """
    vmap can't call a function with many parameters,
    so this function routes it to the algorithm.
    """
    return causal_reduction(batch, "max")

def causal_min_reduction(batch):
    """
    vmap can't call a function with many parameters,
    so this function routes it to the algorithm.
    """
    return causal_reduction(batch, "min")

def causal_reduction(batch, function_type):
  """
  This function is a simplification of the causal attention as a static reduction.
  The first token can only attend to itself and remains unchanged.
  Every following token is compared with the previous one.
  """
  sequence_length, token_dimension = batch.size()
  if sequence_length == 1: # there is only one unchanged token
    return batch
  # take the unchanged first token and unsqueeze it into a single tensor
  first_sequence = batch[0].unsqueeze(0)
  # Concatenate the first sequence token with an empty batch
  static_reduction = torch.concat((first_sequence, torch.zeros(sequence_length-1, token_dimension)),0)

  # reduce each token with the previous one
  for token_number in range(1, sequence_length):
    if function_type == "min":
        static_reduction[token_number] = torch.min(batch[token_number], batch[token_number-1])
    elif function_type == "mean":
        static_reduction[token_number] = torch.mean(batch[token_number], batch[token_number-1])
    else: # function_type == "max":
        static_reduction[token_number] = torch.max(batch[token_number], batch[token_number-1])
  return static_reduction

def causal_matrix_reduction(batch):
  """
  This function is a simplification of the causal attention as a static reduction.
  The first token can only attend to itself and remains unchanged.
  Every following token is compared with the previous one.
  """
  sequence_length, token_dimension = batch.size()

  # take the unchanged first token and unsqueeze it into a single tensor
  first_sequence = batch[0].unsqueeze(0)
  # Concatenate the first sequence token with an empty batch
  causal_shift = torch.concat((first_sequence, batch[:-1]),0)
  # stack of the original tensor with the causal shift and the global mean
  concatenation = torch.stack((batch, causal_shift),1)
  # reduce the doubled tensor by the minimum
  static_reduction = torch.min(concatenation, dim=1)

  # only return the min values without the index
  return static_reduction[0]
