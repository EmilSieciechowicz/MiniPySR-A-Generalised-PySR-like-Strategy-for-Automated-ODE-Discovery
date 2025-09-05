import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torchdiffeq import odeint # type: ignore

from IPython.display import display
import json
import gzip

from tqdm import tqdm
import warnings
from numpy.lib.stride_tricks import sliding_window_view

import Function_Generation
import Evaluation
import Integrators


def mut_2_parallel(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """
    Case 2 mutation in Algorithm 3 of PySR (mutate operator). 
    Replaces a binary operator with a random binary operator and unary operator with a random unary operator.
    Runs in parallel by using tensor operations in pytorch
    Inputs:
        token_seq: (torch.tensor) token sequence of the expressions to mutate. 
                            Padded prior to inputting.
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
    Outputs:
        new_token_seq: (torch.tensor) mutated random token sequences
    """

    # Masks of operators and corresponding idxs in the token sequence
    binary_mask = torch.isin(token_seq, binary_tokens)
    unary_mask  = torch.isin(token_seq, unary_tokens)

    # Generate random tensors
    binary_mut = torch.randint(binary_tokens[0], binary_tokens[-1]+1, size=token_seq.size(), device=GPU_device)
    unary_mut  = torch.randint(unary_tokens[0],  unary_tokens[-1]+1,  size=token_seq.size(), device=GPU_device)

    # Modify relevant tokens mutates all of them currently
    binary_or_unary = torch.randint(0,2, size=torch.Size([1]), device=GPU_device)
    
    mutated_token_seq = torch.where(binary_or_unary == 0, 
                                torch.where(binary_mask, binary_mut, token_seq),
                                torch.where(unary_mask, unary_mut,  token_seq),)
    
    # # checks for token sequences that give expressions like (x-x = 0)
    # value_storage_seq = torch.ones_like(token_seq, device=GPU_device).float()*-1

    # zero_check1 = Evaluation.evaluate_tokens(token_seq, value_storage_seq,
    #                             binary_tokens, unary_tokens,leaf_tokens, None_token, 
    #                             GPU_device, integrator_data=torch.tensor([torch.pi + torch.e]))
    
    # zero_check2 = Evaluation.evaluate_tokens(token_seq, value_storage_seq,
    #                             binary_tokens, unary_tokens,leaf_tokens, None_token, 
    #                             GPU_device, integrator_data=torch.tensor([torch.pi + torch.e + 1]))
    
    # mutated_token_seq = torch.where(abs(zero_check2 - zero_check1) < 1e-9 , token_seq, mutated_token_seq)

    return mutated_token_seq
batched_mut_2_parallel = torch.vmap(mut_2_parallel, in_dims=(0,None,None,None,None,None), randomness='different')


def identify_subtrees_old2(token_seq,  binary_tokens, unary_tokens, leaf_tokens, None_token):
    """
    Locates binary subtrees ([binary_operator, leaf, leaf]) or unary subtree ([unary_operator, leaf, None]), in a
    token sequence. Partitions a sequence into a elements of 3 and checks for the above structure.
    To be expanded to check for subtrees greater than complexity 3.
    Inputs:
        token_seqs_padded: (torch.tensor) token sequences of the expressions to mutate
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
    Outputs:
        subtrees_start_idx: (torch.tensor) tensor containing the index of the parent nodes of each subtree 
                            [token_sequence_index, parent_token_idx]
    """

    # tensor holding all the different sequences of 3
    token_windows = token_seq.unfold(dimension=-1, size=3, step=1)


    # Creating masks (only for end subtree)
    is_binary_parent_node    = torch.isin(token_windows[:,0], binary_tokens)
    is_binary_leaf_node_left = torch.isin(token_windows[:,1], leaf_tokens) | (token_windows[:,1] <= -10)
    is_binary_leaf_node_right = torch.isin(token_windows[:,2], leaf_tokens) | (token_windows[:,2] <= -10)
    is_binary_subtree = is_binary_parent_node&is_binary_leaf_node_left&is_binary_leaf_node_right

    is_unary_parent_node    = torch.isin(token_windows[:,0], unary_tokens)
    is_unary_leaf_node_left = torch.isin(token_windows[:,1], leaf_tokens) | (token_windows[:,1] <= -10)
    is_unary_leaf_node_right = torch.isin(token_windows[:,2], None_token)
    is_unary_subtree = is_unary_parent_node&is_unary_leaf_node_left&is_unary_leaf_node_right

    is_subtree = is_binary_subtree | is_unary_subtree

    # Indexes of the subtrees in token_seq
    subtrees_start_idx = torch.stack(torch.where(is_subtree)).T
    
    return subtrees_start_idx.squeeze(1)


def identify_padding_vmapped(token_seq, GPU_device):
    """
    Takes a token sequence and returns the index of the first pad token (or one above the max index if no pad tokens).
    (Currently redundant in Mini-ODE-PySR)
    Inputs:
        token_seq: (torch.tensor) single token sequence, shape = (n,)

    Outputs:
        padded_start_idx: (torch.tensor) index of the first token that is a padding element
    """
    padding_mask = token_seq == -1
    idxs = torch.arange(token_seq.shape[0], device=GPU_device)

    # Set index to last index + 1 if not padded
    padded_idxs = torch.where(padding_mask, idxs, torch.full_like(idxs, token_seq.shape[0], device=GPU_device))
    padded_start_idx = padded_idxs.min().unsqueeze(dim = 0) 
    return padded_start_idx


def mut_4_parallel(token_seq, token_seq_subtrees_start_idxs, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """"
    Returns a token sequence where either a binary node or unary node is inserted above a subtree of complexity 3. 
    Function works in sequence for the main loop with some masking here and there.
    Inputs:
        token_seq: (torch.tensor) token sequence of the expression to mutate
        token_seq_subtrees_start_idx: (torch.tensor) padded tensor witht he idxs of the subtree parent nodes
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
    Outputs:
        mutated_token_seq_out: (torch.tensor) mutated token sequence
    """
    
    # uniform_dist = torch.distributions.Uniform(0,1)
    # binary_or_unary = uniform_dist.sample(sample_shape = torch.Size([1])).to(GPU_device)
    binary_or_unary = torch.rand(1, device=GPU_device)

    binary_prob = 0.8
    unary_prob  = 1 - binary_prob


    subtree_idx_storage__pad_start = identify_padding_vmapped(token_seq_subtrees_start_idxs, GPU_device) # Stores where padding starts in the subtree idx
    # print(subtree_idx_storage__pad_start)

    # turns padded tokens into one of the ones that inot a subtree start idx (iteraties forward)
    for i in range(token_seq_subtrees_start_idxs.shape[0]):
        token_seq_subtrees_start_idxs[i] = torch.where(token_seq_subtrees_start_idxs[i] == -1,
                                                        token_seq_subtrees_start_idxs[i-subtree_idx_storage__pad_start], token_seq_subtrees_start_idxs[i])

    idx_choice           = torch.randint(0,token_seq_subtrees_start_idxs.shape[0], size=torch.Size([1]), device=GPU_device)
    where_to_add         = token_seq_subtrees_start_idxs[idx_choice]
    where_to_add_flipped = (token_seq.shape[0] - 1) - where_to_add # corresponding index for the flipped token sequence

    new_binary_token    = torch.randint(binary_tokens[0],   binary_tokens[-1]+1,    size=torch.Size([1]), device=GPU_device)
    new_unary_token     = torch.randint(unary_tokens[0],    unary_tokens[-1]+1,     size=torch.Size([1]), device=GPU_device)
    new_leaf_token      = torch.randint(leaf_tokens[0],     leaf_tokens[-1]+1,      size=torch.Size([1]), device=GPU_device)

    # traverse backwards 
    mutated_token_seq_flipped = torch.flip(token_seq,[0])
    mutated_token_seq_flipped_ext = torch.cat([mutated_token_seq_flipped, torch.tensor([-1])]) # need extra element to get rid of error when accessing [i+2] otu of bounds

    for i in range(token_seq.shape[0]-1):
        mutated_token_seq_flipped_ext[i] = torch.where(i <= where_to_add_flipped-4, mutated_token_seq_flipped_ext[i+2],
                        torch.where(i<=where_to_add_flipped, mutated_token_seq_flipped_ext[i+1], mutated_token_seq_flipped_ext[i]))
        
    for i in range(token_seq.shape[0]):
        mutated_token_seq_flipped[i] = mutated_token_seq_flipped_ext[i]

    mutated_token_seq                   = torch.cat([torch.flip(mutated_token_seq_flipped,[0]),torch.tensor([-1,-1], device=GPU_device)])
    mutated_token_seq[where_to_add]     = torch.where(binary_or_unary<=binary_prob, new_binary_token, new_unary_token)
    mutated_token_seq[where_to_add+4]   = torch.where(binary_or_unary<=binary_prob, new_leaf_token, None_token)

    mutated_token_seq_out = torch.full_like(token_seq, fill_value=-1, device=GPU_device)
    for i in range(token_seq.shape[0]):
        mutated_token_seq_out[i] = mutated_token_seq[i]

    return mutated_token_seq_out

def mut_4_maxed_token_seq(token_seq, token_seq_subtrees_start_idxs, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """"
    Same as parallel but also checks if token sequnce has the maximum number of tokens.
    Inputs:
        token_seq: (torch.tensor) token sequence of the expression to mutate
        token_seq_subtrees_start_idx: (torch.tensor) padded tensor witht he idxs of the subtree parent nodes
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
    Outputs:
        token_seq: (torch.tensor) mutated token sequence
    """

    mutated_token_seq = torch.where(token_seq[-1] != -1, 
                                    token_seq, mut_4_parallel(torch.clone(token_seq), token_seq_subtrees_start_idxs, 
                                                              binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device))
    

    # value_storage_seq = torch.ones_like(token_seq, device=GPU_device).float()*-1
    # zero_check = Evaluation.evaluate_tokens(token_seq, value_storage_seq,
    #                             binary_tokens, unary_tokens,leaf_tokens, None_token, 
    #                             GPU_device, integrator_data=torch.tensor([torch.pi + torch.e]))
    # mutated_token_seq = torch.where(zero_check == 0, token_seq, mutated_token_seq)
    
    return mutated_token_seq
batched_mut_4 = torch.vmap(mut_4_maxed_token_seq, in_dims=(0,0,None,None,None,None,None), randomness='different')


def mut_5_parallel(token_seq, token_seq_subtrees_start_idxs, leaf_tokens, GPU_device):
    """"
    Deletes a complexity 3 subtree from a random part of the token sequence
    Inputs:
        token_seq: (torch.tensor) token sequence of the expression to mutate
        token_seq_subtrees_start_idx: (torch.tensor) padded tensor witht he idxs of the subtree parent nodes
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
    Outputs:
        token_seq: (torch.tensor) mutated token sequence
    """

    subtree_idx_storage__pad_start = identify_padding_vmapped(token_seq_subtrees_start_idxs, GPU_device)
        
    for i in range(token_seq_subtrees_start_idxs.shape[0]):
        token_seq_subtrees_start_idxs[i] = torch.where(token_seq_subtrees_start_idxs[i] == -1,
                                                        token_seq_subtrees_start_idxs[i-subtree_idx_storage__pad_start], token_seq_subtrees_start_idxs[i])


    idx_choice           = torch.randint(0,token_seq_subtrees_start_idxs.shape[0], size=torch.Size([1]), device=GPU_device)
    where_to_add         = token_seq_subtrees_start_idxs[idx_choice]

    # traverse forwards
    for i in range(token_seq.shape[0]-2):
        token_seq[i] = torch.where(i > where_to_add, token_seq[i+2], token_seq[i])
    
    token_seq[where_to_add] = leaf_tokens
    token_seq[token_seq.shape[0]-2] = -1
    token_seq[token_seq.shape[0]-1] = -1
    return token_seq

def mut_5_cmplx_3_case(token_seq, token_seq_subtrees_start_idxs, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """"
    Deletes a complexity 3 subtree from a random part of the token sequence and ensures that if only a complexity 3 tree remains it is not deleted.
    Inputs:
        token_seq: (torch.tensor) token sequence of the expression to mutate
        token_seq_subtrees_start_idx: (torch.tensor) padded tensor witht he idxs of the subtree parent nodes
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
    Outputs:
        token_seq: (torch.tensor) mutated token sequence
    """

    mutated_token_seq = torch.where(token_seq[3] == -1, 
                                    token_seq, mut_5_parallel(torch.clone(token_seq), torch.clone(token_seq_subtrees_start_idxs), leaf_tokens, GPU_device))

    # value_storage_seq = torch.ones_like(token_seq, device=GPU_device).float()*-1

    # zero_check = Evaluation.evaluate_tokens(token_seq, value_storage_seq,
    #                             binary_tokens, unary_tokens,leaf_tokens, None_token, 
    #                             GPU_device, integrator_data=torch.tensor([torch.pi + torch.e]))
    
    # mutated_token_seq = torch.where(zero_check == 0, token_seq, mutated_token_seq)

    return mutated_token_seq
batched_mut_5 = torch.vmap(mut_5_cmplx_3_case, in_dims=(0,0,None,None,None,None,None), randomness='different')


def mutate_operator(token_seqs_padded, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """"
    Takes an expression and creates a copy before mutating it.
    Inputs:
        token_seqs_padded: (torch.tensor) token sequences of the expressions to mutate (copied in function)
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
        print_op = (Boolean) print operations statements
    Output:
        mutated_token_seqs = (torch.tensor) mutated token sequences. Same first dimension as input
    """""
    # Token sequence
    token_seqs_padded_copy = torch.clone(token_seqs_padded)


    # Which mutation to do
    mut_type = torch.randint(0, 4, size=torch.Size([1]), device=GPU_device)

    token_seqs_subtrees_start_idxs = [identify_subtrees_old2(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token) for token_seq in token_seqs_padded]
    token_seqs_subtrees_start_idxs = torch.nn.utils.rnn.pad_sequence(token_seqs_subtrees_start_idxs, batch_first=True, padding_value=-1)

    # Case 2
    if mut_type == 0:
        mutated_token_seqs = batched_mut_2_parallel(token_seqs_padded_copy, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device)
        # print('case2')
        # print(mutated_token_seqs)
    
    # Case 4
    elif mut_type == 1:
        mutated_token_seqs = batched_mut_4(token_seqs_padded_copy, token_seqs_subtrees_start_idxs, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device)
        # print('case4')
        # print(mutated_token_seqs)

     # Case 5
    elif mut_type == 2:
        mutated_token_seqs = batched_mut_5(token_seqs_padded_copy, token_seqs_subtrees_start_idxs, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device)
        # print('case5')
        # print(mutated_token_seqs)

    # Case 8 
    else:
        mutated_token_seqs = token_seqs_padded_copy
        # print('case8')
        # print(mutated_token_seqs)
    
    assert mutated_token_seqs.size()[0] == token_seqs_padded.size()[0]      # Checks that the number of expressions stays the same

    #  Maybe add filter later on to check for feasibility

    return mutated_token_seqs


def crossover_parallel(token_seq1, token_seq2, subtrees_start_idx1, subtrees_start_idx2, GPU_device):
    """"
    Performs the crossover operation between randomly selected pairs of padded token sequences
    Inputs:
        token_seqs_padded: (torch.tensor) all the padded token sequences to be crossed over (current population)
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
    Outputs:
        crossed_token_seqs: (torch.tensor) tensor containing all the crossed token sequences
    """

    # print(f"tok_seq1:{token_seq1}")
    # print(f"tok_seq2:{token_seq2}")
    # print(f"sub_start1:{subtrees_start_idx1}")
    # print(f"sub_start2:{subtrees_start_idx2}")

    subtree_idx_storage__pad_start1 = identify_padding_vmapped(subtrees_start_idx1, GPU_device)
    subtree_idx_storage__pad_start2 = identify_padding_vmapped(subtrees_start_idx2, GPU_device)
        
    for i in range(subtrees_start_idx1.shape[0]):
        subtrees_start_idx1[i] = torch.where(subtrees_start_idx1[i] == -1,
                                            subtrees_start_idx1[i-subtree_idx_storage__pad_start1], subtrees_start_idx1[i])
        
    for i in range(subtrees_start_idx2.shape[0]):
        subtrees_start_idx2[i] = torch.where(subtrees_start_idx2[i] == -1,
                                            subtrees_start_idx2[i-subtree_idx_storage__pad_start2], subtrees_start_idx2[i])
        
    idx_choice1     = torch.randint(0,subtrees_start_idx1.shape[0], size=torch.Size([1]), device=GPU_device)
    idx_choice2     = torch.randint(0,subtrees_start_idx2.shape[0], size=torch.Size([1]), device=GPU_device)
    which_to_cross1 = subtrees_start_idx1[idx_choice1]
    which_to_cross2 = subtrees_start_idx2[idx_choice2]

    subtree1 = torch.cat([token_seq1[which_to_cross1], token_seq1[which_to_cross1+1], token_seq1[which_to_cross1+2]])
    subtree2 = torch.cat([token_seq2[which_to_cross2], token_seq2[which_to_cross2+1], token_seq2[which_to_cross2+2]])

    token_seq1[which_to_cross1], token_seq1[which_to_cross1+1], token_seq1[which_to_cross1+2] = subtree2[0], subtree2[1], subtree2[2]
    token_seq2[which_to_cross2], token_seq2[which_to_cross2+1], token_seq2[which_to_cross2+2] = subtree1[0], subtree1[1], subtree1[2]

    crossed_token_seqs = torch.stack([token_seq1, token_seq2])

    return crossed_token_seqs
batched_crossover = torch.vmap(crossover_parallel,in_dims=(0,0,0,0,None), randomness='different')

