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

import Search
import Integrators


# Token evaluation

def identify_last_subtree(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device):
    """"
    Takes a token sequence and returns the index of the last subtree that appears
    Inputs:
         token_seq: (torch.tensor) single token sequence, shape = (n,)
         binary_tokens: (torch.tensor) tensor containing all binary operator tokens.
         unary_tokens: (torch.tensor) tensor containing all unary operator tokens
         leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
         None_token: (torch.tensor) tensor containing the None token.
     Outputs:
        last_subtree_idx: (torch.tensor) tensor containing the index of the parent token of the first subtree in the sequence, 
                    else the index one above tree lenght is returned
    """

    # tensor holding all the different sequences of 3
    token_windows = torch.clone(token_seq.unsqueeze(dim = 0)).unfold(dimension=-1, size=3, step=1)
    # padding to keep consistent sizes for where condition
    token_windows = F.pad(token_windows, pad=(0,0,0,2), value=-1)

    # Creating masks (only for end subtree)
    is_binary_parent_node       = torch.isin(token_windows[:,:,0], binary_tokens)
    is_binary_leaf_node_left    = torch.isin(token_windows[:,:,1], leaf_tokens) | (token_windows[:,:,1] <= -10)
    is_binary_leaf_node_right   = torch.isin(token_windows[:,:,2], leaf_tokens) | (token_windows[:,:,2] <= -10)
    is_binary_subtree = is_binary_parent_node&is_binary_leaf_node_left&is_binary_leaf_node_right

    is_unary_parent_node        = torch.isin(token_windows[:,:,0], unary_tokens)
    is_unary_leaf_node_left     = torch.isin(token_windows[:,:,1], leaf_tokens) | (token_windows[:,:,1] <= -10)
    is_unary_leaf_node_right    = torch.isin(token_windows[:,:,2], None_token)
    is_unary_subtree = is_unary_parent_node&is_unary_leaf_node_left&is_unary_leaf_node_right

    is_subtree = is_binary_subtree | is_unary_subtree
    is_subtree = is_subtree.squeeze()

    idxs = torch.arange(token_seq.shape[0], device=GPU_device)

    # Indexes of the subtrees in token_seq
    # subtrees_start_idx = torch.where(is_subtree, idxs, torch.full_like(idxs, token_seq.shape[0]))
    subtrees_start_idx = torch.where(is_subtree, idxs, torch.full_like(idxs, 0, device=GPU_device))

    # first_subtree_idx = subtrees_start_idx.min().unsqueeze(dim = 0) 
    last_subtree_idx = subtrees_start_idx.max().unsqueeze(dim = 0) 
    
    return last_subtree_idx


def eval_cmplx_3_subtree(subtree, token_seq, value_storage_seq, leaf_tokens, GPU_device, integrator_data):
    """"
    Evaluates the complexity 3 subtree. due to vmapping constraints all outcomes are calculated. 
    Inputs:
        subtree: (torch.tensor) tensor with 3 tokens that evaluates a complexity 3 subtree
        token_seq: (torch.tensor) single token sequence, shape = (n,)
        value_storage_seq: (torch.tensor) tensor of same shape as token_seq, contains evaluated subtree values
                                            at corresponding indexes.
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        integrator_data: (torch.tensor) point at which to evaluate the function generated. Single point at which evaluation occurs at.
    Outputs:
        evaluation: (torch.tensor) values of the subtree given the input data or previous suntree evaluation.
    """

    # assert (subtree[0] in binary_tokens)|((subtree[0] in unary_tokens))

    token1 = subtree[0]
    token2 = subtree[1]
    token3 = subtree[2]

    idxs = torch.arange(token_seq.shape[0],device=GPU_device)

    # If leaf location does not matter, only matters for unique id (fetching precomputed values)
    location2 = torch.where(token_seq == token2, idxs, torch.full_like(token_seq, -1, device=GPU_device)).max().unsqueeze(dim=0)
    location3 = torch.where(token_seq == token3, idxs, torch.full_like(token_seq, -1, device=GPU_device)).max().unsqueeze(dim=0)

    # Adding cases
    cond_add1       = torch.isin(token1,torch.tensor(0, device=GPU_device)) & torch.isin(token2, leaf_tokens) & torch.isin(token3, leaf_tokens)
    cond_add1_val   = torch.add(integrator_data, integrator_data)
    cond_add2       = torch.isin(token1,torch.tensor(0, device=GPU_device)) & torch.isin(token2, leaf_tokens) & (token3 <= -10)
    cond_add2_val   = torch.add(integrator_data, value_storage_seq[location3])
    cond_add3       = torch.isin(token1,torch.tensor(0, device=GPU_device)) & (token2 <= -10) & torch.isin(token3, leaf_tokens)
    cond_add3_val   = torch.add(value_storage_seq[location2], integrator_data)
    cond_add4       = torch.isin(token1,torch.tensor(0, device=GPU_device)) & (token2 <= -10) & (token3 <= -10)
    cond_add4_val   = torch.add(value_storage_seq[location2], value_storage_seq[location3])

    # Subtracting cases
    cond_sub1       = torch.isin(token1,torch.tensor(1, device=GPU_device)) & torch.isin(token2, leaf_tokens) & torch.isin(token3, leaf_tokens)
    cond_sub1_val   = torch.sub(integrator_data, integrator_data)
    cond_sub2       = torch.isin(token1,torch.tensor(1, device=GPU_device)) & torch.isin(token2, leaf_tokens) & (token3 <= -10)
    cond_sub2_val   = torch.sub(integrator_data, value_storage_seq[location3])
    cond_sub3       = torch.isin(token1,torch.tensor(1, device=GPU_device)) & (token2 <= -10) & torch.isin(token3, leaf_tokens)
    cond_sub3_val   = torch.sub(value_storage_seq[location2], integrator_data)
    cond_sub4       = torch.isin(token1,torch.tensor(1, device=GPU_device)) & (token2 <= -10) & (token3 <= -10)
    cond_sub4_val   = torch.sub(value_storage_seq[location2], value_storage_seq[location3])
    
    # Multiplication cases
    cond_mul1       = torch.isin(token1,torch.tensor(2, device=GPU_device)) & torch.isin(token2, leaf_tokens) & torch.isin(token3, leaf_tokens)
    cond_mul1_val   = torch.mul(integrator_data, integrator_data)
    cond_mul2       = torch.isin(token1,torch.tensor(2, device=GPU_device)) & torch.isin(token2, leaf_tokens) & (token3 <= -10)
    cond_mul2_val   = torch.mul(integrator_data, value_storage_seq[location3])
    cond_mul3       = torch.isin(token1,torch.tensor(2, device=GPU_device)) & (token2 <= -10) & torch.isin(token3, leaf_tokens)
    cond_mul3_val   = torch.mul(value_storage_seq[location2], integrator_data)
    cond_mul4       = torch.isin(token1,torch.tensor(2, device=GPU_device)) & (token2 <= -10) & (token3 <= -10)
    cond_mul4_val   = torch.mul(value_storage_seq[location2], value_storage_seq[location3])

    # Division cases
    cond_div1       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & torch.isin(token2, leaf_tokens) & torch.isin(token3, leaf_tokens)
    cond_div1_val   = torch.div(integrator_data, integrator_data)
    cond_div2       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & torch.isin(token2, leaf_tokens) & (token3 <= -10)
    cond_div2_val   = torch.div(integrator_data, value_storage_seq[location3])
    cond_div3       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & (token2 <= -10) & torch.isin(token3, leaf_tokens)
    cond_div3_val   = torch.div(value_storage_seq[location2], integrator_data)
    cond_div4       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & (token2 <= -10) & (token3 <= -10)
    cond_div4_val   = torch.div(value_storage_seq[location2], value_storage_seq[location3])

    # # abs cases
    # cond_abs1       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & torch.isin(token2, leaf_tokens)
    # cond_abs1_val   = torch.abs(integrator_data)
    # cond_abs2       = torch.isin(token1,torch.tensor(3, device=GPU_device)) & (token2 < -10)
    # cond_abs2_val   = torch.abs(value_storage_seq[location2])

    # sin cases
    cond_sin1       = torch.isin(token1,torch.tensor(4, device=GPU_device)) & torch.isin(token2, leaf_tokens)
    cond_sin1_val   = torch.sin(integrator_data)
    cond_sin2       = torch.isin(token1,torch.tensor(4, device=GPU_device)) & (token2 <= -10)
    cond_sin2_val   = torch.sin(value_storage_seq[location2])

    # # tan cases
    # cond_tan1       = torch.isin(token1,torch.tensor(5, device=GPU_device)) & torch.isin(token2, leaf_tokens)
    # cond_tan1_val   = torch.tan(integrator_data)
    # cond_tan2       = torch.isin(token1,torch.tensor(5, device=GPU_device)) & (token2 < -10)
    # cond_tan2_val   = torch.tan(value_storage_seq[location2])

    # exp cases
    cond_exp1       = torch.isin(token1,torch.tensor(5, device=GPU_device)) & torch.isin(token2, leaf_tokens)
    cond_exp1_val   = torch.exp(integrator_data)
    cond_exp2       = torch.isin(token1,torch.tensor(5, device=GPU_device)) & (token2 <= -10)
    cond_exp2_val   = torch.exp(value_storage_seq[location2])

    # Generalised PySR (would need to include leaf parameter to extend to multiple inputs)
    evaluation =    torch.where(cond_add1, cond_add1_val,
                    torch.where(cond_add2, cond_add2_val,
                    torch.where(cond_add3, cond_add3_val,
                    torch.where(cond_add4, cond_add4_val,
                    torch.where(cond_sub1, cond_sub1_val,
                    torch.where(cond_sub2, cond_sub2_val,
                    torch.where(cond_sub3, cond_sub3_val,
                    torch.where(cond_sub4, cond_sub4_val,
                    torch.where(cond_mul1, cond_mul1_val,
                    torch.where(cond_mul2, cond_mul2_val,
                    torch.where(cond_mul3, cond_mul3_val,
                    torch.where(cond_mul4, cond_mul4_val,
                    torch.where(cond_div1, cond_div1_val,
                    torch.where(cond_div2, cond_div2_val,
                    torch.where(cond_div3, cond_div3_val,
                    torch.where(cond_div4, cond_div4_val,
                    torch.where(cond_sin1, cond_sin1_val,
                    torch.where(cond_sin2, cond_sin2_val,
                    torch.where(cond_exp1, cond_exp1_val,
                    torch.where(cond_exp2, cond_exp2_val, 
                        torch.tensor([69420], device=GPU_device)))))))))))))))))))))


    return evaluation


def evaluate_subtrees_once(token_seq, value_storage_seq, subtree_start_idx, leaf_tokens, GPU_device, integrator_data):
    """"
    Evaluates a subtree in a padded token sequence. Replaces the parent token of that subtree with a unique id and
    shifts the other elemets two spaces to the left. The evaluation value is stored in value_storage_seq at the same index
    as the unique id. value_storage_seq is updated in the same way as token_seq in regards to elements shifting.
    Inputs:
        token_seq: (torch.tensor) single token sequence, shape = (n,)
        value_storage_seq: (torch.tensor) tensor of same shape as token_seq, contains evaluated subtree values at corresponding indexes.
        subtree_start_idx: (torch.tensor) tensor containing the index of the parent node of the subtree to be evaluated
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        integrator_data: (torch.tensor) point at which to evaluate the function generated. Single point at which evaluation occurs at.
    Outputs:
        token_seq: (torch.tensor) single token sequence, shape = (n,) (updated with unique ids')
    """

    subtree     = torch.cat([token_seq[subtree_start_idx], token_seq[subtree_start_idx+1], token_seq[subtree_start_idx+2]])
    token1      = subtree[0]
    unique_id   = -torch.randint(int(10e5), size=(1,), device=GPU_device) - 10
    value_storage_seq[subtree_start_idx]       = torch.where(token1 <= -10 ,
                                value_storage_seq[subtree_start_idx],
                                eval_cmplx_3_subtree(subtree,token_seq, value_storage_seq, leaf_tokens, GPU_device, integrator_data))


    # token_seq[subtree_start_idx] = unique_id        # will probably have to edit this line for the final corner case

    token_seq[subtree_start_idx] = torch.where(token1 <= -10, token_seq[subtree_start_idx], unique_id)

    for i in range(token_seq.shape[0]-2):
        token_seq[i]            = torch.where(i > subtree_start_idx, token_seq[i+2], token_seq[i])
        value_storage_seq[i]    = torch.where(i > subtree_start_idx, value_storage_seq[i+2], value_storage_seq[i])
    
    token_seq[token_seq.shape[0]-2] = -1
    token_seq[token_seq.shape[0]-1] = -1
    value_storage_seq[token_seq.shape[0]-2] = -1
    value_storage_seq[token_seq.shape[0]-1] = -1
    
    return token_seq


def evaluate_tokens(token_seq, value_storage_seq, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, integrator_data):
    """
    Evaluates the whole sequence of tokens iteratively and returns a tensor containing the evaluation.
    (values updated in value_storage_seq)
    Inputs:
        token_seq: (torch.tensor) single token sequence, shape = (n,)
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
        integrator_data: (torch.tensor) point at which to evaluate the function generated. Single point at which evaluation occurs at.
    Outputs:
        preds: (torch.tensor) tensor of predictions for every expression in token_seqs_padded (each column corresponds to a token sequence)
    """

    # Main loop of the function (evaluates subtree once per iteration, does minimum number of iterations possible for given sequence)
    for i in range(int(np.floor(token_seq.shape[0])/2)):
        subtree_start_idx  = identify_last_subtree(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device)
        token_seq  = evaluate_subtrees_once(token_seq, value_storage_seq, subtree_start_idx, leaf_tokens, GPU_device, integrator_data)

    pred = value_storage_seq[0]

    return pred

# Losses
def complexity_eval(token_seq):
    """"
    Calculates a the complexity of a given expression. (Counting None tokens helps prevent bias towards unary tokens)
    Inputs:
        token_seq: (torch.tensor) single token sequence, shape = (n,)
    Outputs:
        complexity: (torch.tensor) tensor of integers representing the complexity of each token seq
    """
    complexity = (token_seq > -1).sum().unsqueeze(0)

    return complexity

def Custom_Loss_function_space(token_seqs_padded, data, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, alpha = 0.01, beta = 0.2, gamma = 1):
    """
    Calculated the loss function of an expression given data. Balances complexity and accuracy.
    In this case a simple MSE vs complexity balance.
    Inputs:
        token_seqs_padded: (torch.tensor) all token sequences which need loss calculation (padded with -1 afer end of token sequence)
        data: (torch.tensor) data from experiment or other. ([x, f(x)])
        torch_operator_space: (dict) mapping to torch operations rather than sympy purely for evaluation.
                                also contains the mappings of calculated subtrees.
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves: (tuple of str) Set of allowable leaf values (input variables).
        alpha: (float) constant which determines how much complexity is penalised
        beta: (float) constant which determines how much the same number of expressions is penalised
    Outputs:
        custom_losses: (float) acc_loss(E)*exp(frequency(E) * complexity(E))
    """
    # Accuracy losses
    targets = torch.unsqueeze(data[:,-1],1)
    evaluation_points   = data[:,0]
    preds = (torch.ones_like(evaluation_points,dtype=torch.float,device=GPU_device)*-1).unsqueeze(0).repeat(token_seqs_padded.shape[0],1).mT

    value_storage_seqs  = torch.ones_like(token_seqs_padded,dtype=torch.float,device=GPU_device)*-1                   # tracks values for all sequences
    value_storage_seqs_all = value_storage_seqs.unsqueeze(0).repeat(evaluation_points.shape[0],1,1) # tracks values for all sequences for all points

    # Nested batching for efficiency
    batched_seqs_evaluate_tokens    = torch.vmap(evaluate_tokens,               in_dims=(0, 0, None, None, None, None, None, None), randomness='same')
    batched_points_evaluate_tokens  = torch.vmap(batched_seqs_evaluate_tokens,  in_dims=(None, 0, None, None, None, None, None, 0), randomness='same')
    preds = batched_points_evaluate_tokens(torch.clone(token_seqs_padded), value_storage_seqs_all, 
                                           binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, evaluation_points)

    assert targets.shape[0] == preds.shape[0]

    # MSE Loss
    acc_losses = torch.linalg.norm(preds - targets, axis = 0)/targets.shape[0]

    
    # Structure_losses
    factors         = preds/targets
    factors_rolled  = factors.roll(1,dims=0)
    differences     = (factors-factors_rolled).roll(-1,dims=0)
    structure_losses= gamma*abs(differences[:-1]).mean(dim=0)


    # Complexity Losses
    batched_complexity_eval = torch.vmap(complexity_eval)
    node_count = batched_complexity_eval(token_seqs_padded).squeeze()
    cmplx_losses = alpha*node_count


    # Frequency Losses (have to juggle data as unique not implemented fot 'mps')
    unique_token_sequences, inverse_indecies, counts = torch.unique(preds.to('cpu'), return_counts=True, return_inverse=True, dim = 1)

    if GPU_device != 'cpu':
        unique_token_sequences  =  unique_token_sequences.to(GPU_device)
        inverse_indecies        = inverse_indecies.to(GPU_device)
        counts                  = counts.to(GPU_device)               
    frequency_loss = beta*counts[inverse_indecies]


    # # Overall Loss (no structure penalty)
    # custom_losses = acc_losses*torch.exp(frequency_loss*cmplx_losses)

    # Overall Loss (structure penalty included)
    custom_losses         = acc_losses*torch.exp(frequency_loss+cmplx_losses+structure_losses)
    custom_losses_MSE     = acc_losses*torch.exp(torch.tensor(0))
    custom_losses_freq    = acc_losses*torch.exp(frequency_loss)

    # # Overall Loss (structure penalty included harder cutoff maybe put in new function)
    # custom_losses         = acc_losses*torch.exp(frequency_loss+cmplx_losses)*(structure_losses)
    # custom_losses_no_freq = acc_losses*torch.exp(cmplx_losses)*(structure_losses)


    # Add assertion for custom losses 
    return custom_losses, custom_losses_MSE, custom_losses_freq, counts[inverse_indecies], node_count,  abs(differences[:-1]).mean(dim=0)



def Custom_Loss_static_model_space(token_seqs_padded, data, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, alpha = 0.01, beta = 0.2, gamma = 2):
    """
    Calculated the loss function of all expressions given data. Balances complexity,accuracu and frequencey of appeard token sequence.
    In this case a simple MSE vs complexity balance.
    Inputs:
        token_seqs_padded: (torch.tensor) all token sequences which need loss calculation (padded with -1 afer end of token sequence)
        data: (torch.tensor) data from experiment or other. ([x, f(x)])
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
        alpha: (float) constant which determines how much complexity is penalised
        beta: (float) constant which determines how much the same number of expressions is penalised
    Outputs:
        custom_losses: (float) acc_loss(E)*exp(frequency(E) * complexity(E))
    """
    # Accuracy losses
    targets             = torch.unsqueeze(data[:,-1],1)
    initial_condition   = targets[0]
    evaluation_points   = data[:,0]

    value_storage_seqs  = torch.ones_like(token_seqs_padded,dtype=torch.float, device=GPU_device)*-1                   # tracks values for all sequences

    evaluation_sols_all = (torch.ones_like(evaluation_points,dtype=torch.float, device=GPU_device)*-1).unsqueeze(0).repeat(token_seqs_padded.shape[0],1)

    def static_integrate_token_seq(token_seq, value_storage_seq, evaluation_sols, initial_condition, evaluation_points, binary_tokens, unary_tokens, leaf_tokens, None_token):

        def candidate(x):
            du_dx = evaluate_tokens(torch.clone(token_seq), value_storage_seq, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, integrator_data = x)
            return du_dx
            
        pred = Integrators.RK4_static_integrator(candidate, evaluation_sols, initial_condition, evaluation_points, GPU_device)

        return pred
    
    batched_static_integration = torch.func.vmap(static_integrate_token_seq, in_dims=(0,0,0,None,None,None,None,None,None), randomness='same')
    preds = batched_static_integration(token_seqs_padded,value_storage_seqs, evaluation_sols_all, 
                                       initial_condition, evaluation_points, binary_tokens, unary_tokens, leaf_tokens, None_token).mT

    assert targets.shape[0] == preds.shape[0]


    # MSE Loss
    acc_losses = torch.linalg.norm(preds - targets, dim = 0)/targets.shape[0]

    # Structure_losses (BE WARY AS TARGETS MAY BE 0 AND THEREFORE STRUCTURE LOSSES WILL FAIL thats why eps added but not stress tested)
    eps = torch.where(min(targets) == 0, 1e-7, min(targets)*1e-7)
    factors         = preds/(targets+eps)
    factors_rolled  = factors.roll(1,dims=0)
    differences     = (factors-factors_rolled).roll(-1,dims=0)
    structure_losses= gamma*abs(differences[:-1]).mean(dim=0)

    # Complexity Losses
    batched_complexity_eval = torch.vmap(complexity_eval)
    node_count = batched_complexity_eval(token_seqs_padded).squeeze()
    cmplx_losses = alpha*node_count


    # Frequency Losses

    unique_token_sequences, inverse_indecies, counts = torch.unique(preds, return_counts=True, return_inverse=True, dim = 1)
    frequency_loss = beta*counts[inverse_indecies]

    # # Overall Loss (no structure penalty)
    # custom_losses = acc_losses*torch.exp(frequency_loss*cmplx_losses)

    # Overall Loss (structure penalty included)
    custom_losses         = acc_losses*torch.exp(frequency_loss+cmplx_losses+structure_losses)
    custom_losses_MSE     = acc_losses*torch.exp(torch.tensor(0))
    custom_losses_freq    = acc_losses*torch.exp(frequency_loss)

    # # Overall Loss (structure penalty included harder cutoff maybe put in new function)
    # custom_losses         = acc_losses*torch.exp(frequency_loss+cmplx_losses)*(structure_losses)
    # custom_losses_no_freq = acc_losses*torch.exp(cmplx_losses)*(structure_losses)

    # Add assertion for custom losses length equal to number of expressions

    return custom_losses, custom_losses_MSE, custom_losses_freq, counts[inverse_indecies], node_count,  abs(differences[:-1]).mean(dim=0)


def get_fittest_function_space(token_seqs_padded, data, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, alpha = 0.01, beta = 0.2, gamma = 2, number_of_sols = 1):
    """
    Function returns the best performing function according to the loss_metric argument.
    Inputs: 
        token_seqs_padded: (torch.tensor) all token sequences which need loss calculation (padded with -1 afer end of token sequence)
        data: (torch.tensor) data from experiment or other. ([x, f(x)])
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
        alpha: (float) constant which determines how much complexity is penalised
        beta: (float) constant which determines how much the same number of expressions is penalised
        number_of_sols: (int) the number of fittest functions to return (default = 1)
    Outputs:
        best_token_seqs: (torch.tensor) copy of the best expression in the subset of expressions given.
        best_loss: (torch.tensor) loss for the current expression
    """

    custom_losses, custom_losses_MSE, custom_losses_freq, frequencies, complexities, structures = Custom_Loss_function_space(token_seqs_padded, data, 
                                                   binary_tokens, unary_tokens, leaf_tokens, None_token, 
                                                   GPU_device, alpha = alpha, beta = beta, gamma = gamma)
    
    best_idxs =  torch.argsort(custom_losses)[:number_of_sols]

    best_token_seqs     = token_seqs_padded[best_idxs]
    best_losses         = custom_losses[best_idxs]
    best_losses_MSE     = custom_losses_MSE[best_idxs]
    best_losses_freq    = custom_losses_freq[best_idxs]

    best_frequencies    = frequencies[best_idxs]
    best_complexities   = complexities[best_idxs]
    best_structures     = structures[best_idxs]

    assert best_token_seqs.size() == torch.Size([number_of_sols, token_seqs_padded.size()[1]])

    return best_token_seqs, best_losses_freq, best_losses_MSE, best_frequencies, best_complexities, best_structures

def get_fittest_model_space(token_seqs_padded, data, binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device, alpha = 0.01, beta = 0.2, gamma = 2, number_of_sols = 1):
    """
    Function returns the best performing function according to the loss_metric argument.
    Inputs: 
        token_seqs_padded: (torch.tensor) all token sequences which need loss calculation (padded with -1 afer end of token sequence)
        data: (torch.tensor) data from experiment or other. ([x, f(x)])
        binary_tokens: (torch.tensor) tensor containing all the tokens for binary operators
        unary_tokens: (torch.tensor) tensor containing all the tokens for unary operators
        leaf_tokens: (torch.tensor) tensor containing all leaf tokens (input variables).
        None_token: (torch.tensor) tensor containing the None token.
        alpha: (float) constant which determines how much complexity is penalised
        beta: (float) constant which determines how much the same number of expressions is penalised
        number_of_sols: (int) the number of fittest functions to return (default = 1)
    Outputs:
        best_token_seqs: (torch.tensor) copy of the best expression in the subset of expressions given.
        best_loss: (torch.tensor) loss for the current expression
    """
    custom_losses, custom_losses_MSE, custom_losses_freq, frequencies, complexities, structures = Custom_Loss_static_model_space(token_seqs_padded, data, 
                                                binary_tokens, unary_tokens, leaf_tokens, None_token, 
                                                GPU_device, alpha = alpha, beta = beta, gamma = gamma)
    
    best_idxs =  torch.argsort(custom_losses)[:number_of_sols]

    best_token_seqs     = token_seqs_padded[best_idxs]
    best_losses         = custom_losses[best_idxs]
    best_losses_MSE     = custom_losses_MSE[best_idxs]
    best_losses_freq    = custom_losses_freq[best_idxs]

    best_frequencies    = frequencies[best_idxs]
    best_complexities   = complexities[best_idxs]
    best_structures     = structures[best_idxs]

    assert best_token_seqs.size() == torch.Size([number_of_sols, token_seqs_padded.size()[1]])

    return best_token_seqs, best_losses_freq, best_losses_MSE, best_frequencies, best_complexities, best_structures



def pad_groups(token_seqs_padded, mutated_token_seqs):
    """
    Takes two groups of token sequences, and padds them so that they are the same size. Returns a single tensor of all the token sequences.
    Input:
        token_seqs_padded: (torch.tensor) token sequences of the expressions before mutation (or crossover).
        mutated_token_seqs: (torch.tensor) token sequences of the expressions after mutation.
    Output:
        combined_token_seqs: (torch.tensor) tensor of token sequneces stacked vertically and padded to same length.
    """

    old_token_seqs = list(torch.unbind(torch.clone(token_seqs_padded)))
    new_token_seqs = list(torch.unbind(torch.clone(mutated_token_seqs)))
    old_token_seqs.append(new_token_seqs.pop())
    old_token_seqs = torch.nn.utils.rnn.pad_sequence(old_token_seqs,batch_first=True, padding_value=-1)
    combined_token_seqs = torch.cat((old_token_seqs,torch.stack(new_token_seqs)))

    assert (combined_token_seqs.size()[0] == token_seqs_padded.size()[0] + mutated_token_seqs.size()[0]) and (combined_token_seqs.size()[1] == mutated_token_seqs.size()[1])

    return combined_token_seqs



def evolution_function_space(token_seqs_padded, data, binary_operators, unary_operators, leaves, GPU_device,  population_size = 100, iterations=10, TOL = 0.1, alpha = 0.0, beta = 0.1, gamma = 0, mut_prob = 0.6, print_op = False):
    """
    Evolves the population using mutation and crossover operators (for now just mutation). 
    Discards unfit members of the population and maintains constant population size (by discarding remaining members)
    Evloves without parallelisation. 
    Input: 
        token_seqs_padded: (torch.tensor) tensor containing all token seqences to be evolved (initial population)
        mapping to torch operations rather than sympy purely for evaluation.
                                also contains the mappings of calculated subtrees.
        data: (torch.tensor) (n,2) shaped array of experimanetal data
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
        GPU_device: (str) 'cuda' or 'mps' for GPU accelleration
        population_size: (int) size of population
        number_to_mutate: (int) how many tokens to mutate
        iterations: (int) number of evolution iterations to do
        threshold: (float) the threshold for extreme values (should be set as some (safety factor)*max(abs(data)))
    Outputs
        new_population_tokens: (np.array) array of evolved expressions (in token form)

    """
    top_loss_history = []
    top_loss_MSE_history = []

    top_frequency_history   = []
    top_complexity_history  = []
    top_structure_history   = []

    # Tracking Metrics of all solutions through evolution
    all_loss_MSE_history = []
    all_frequency_history   = []
    all_complexity_history  = []
    all_structure_history   = []

    binary_tokens = torch.arange(len(binary_operators), device=GPU_device)
    unary_tokens  = torch.arange(len(binary_operators), len(binary_operators)+len(unary_operators), device=GPU_device)
    leaf_tokens   = torch.arange(len(binary_operators)+len(unary_operators), len(binary_operators)+len(unary_operators)+len(leaves), device=GPU_device)
    None_token    = torch.arange(len(binary_operators)+len(unary_operators)+len(leaves), len(binary_operators)+len(unary_operators)+len(leaves)+1, device=GPU_device)
    # pad_token     = torch.tensor([-1])
    # tree_token    = torch.tensor([-2])          # Tree token for finding trees that arent complexity 3

    token_seqs_padded   = token_seqs_padded.to(GPU_device)
    data                = data.to(GPU_device)

    for i in tqdm(range(iterations), desc="Evolving", disable = not print_op):
        mutate_or_crossover = torch.rand(1, device=GPU_device)

        if mutate_or_crossover > mut_prob:
            # Mutate and add mutated tokens to general population.
            modified_token_seqs = Search.mutate_operator(torch.clone(token_seqs_padded), binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device)

        else:
            which_pair = torch.randperm(token_seqs_padded.shape[0], device=GPU_device).reshape(int(token_seqs_padded.shape[0]/2),2)

            group1 = token_seqs_padded[which_pair[:,0]]
            group2 = token_seqs_padded[which_pair[:,1]]

            subtrees_start_idxs1 = [Search.identify_subtrees_old2(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token) for token_seq in torch.clone(group1)]
            subtrees_start_idxs1 = torch.nn.utils.rnn.pad_sequence(subtrees_start_idxs1, batch_first=True, padding_value=-1)

            subtrees_start_idxs2 = [Search.identify_subtrees_old2(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token) for token_seq in torch.clone(group2)]
            subtrees_start_idxs2 = torch.nn.utils.rnn.pad_sequence(subtrees_start_idxs2, batch_first=True, padding_value=-1)            

            modified_token_seqs_pre = Search.batched_crossover(torch.clone(group1), torch.clone(group2), torch.clone(subtrees_start_idxs1), torch.clone(subtrees_start_idxs2), GPU_device)
            modified_token_seqs     = torch.cat([modified_token_seqs_pre[:,0,:],modified_token_seqs_pre[:,1,:]])

        # candidate_token_seqs = pad_groups(token_seqs_padded, modified_token_seqs)
    
        # Evaluate fitness
        candidate_token_seqs = torch.cat([token_seqs_padded, modified_token_seqs])

        token_seqs_padded, losses, losses_MSE, frequencies, complexities, structures  = get_fittest_function_space(candidate_token_seqs, data, 
                                                                    binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device,
                                                                    alpha = alpha, beta = beta,gamma=gamma, number_of_sols = population_size)
        

        # if GPU_device != 'cpu':
        #     token_seqs_padded = token_seqs_padded_GPU.to('cpu')
        # else:
        #     token_seqs_padded = token_seqs_padded_GPU

        top_loss_history.append(losses[0].to('cpu'))
        top_loss_MSE_history.append(losses_MSE[0].to('cpu'))
        top_frequency_history.append(frequencies[0].to('cpu'))
        top_complexity_history.append(complexities[0].to('cpu'))
        top_structure_history.append(structures[0].to('cpu'))

        all_loss_MSE_history.append(losses_MSE)
        all_frequency_history.append(frequencies.to('cpu'))
        all_complexity_history.append(complexities.to('cpu'))
        all_structure_history.append(structures.to('cpu'))


        if i == 0:
            best_sol  = token_seqs_padded[0]
            best_loss = losses_MSE[0]

            best_frequency  = frequencies[0]
            best_complexity = complexities[0]
            best_structure  = structures[0]

        elif i >= 1:
            if top_loss_MSE_history[-1] < best_loss:
                best_sol  = token_seqs_padded[0]
                best_loss = losses_MSE[0]

                best_frequency  = frequencies[0]
                best_complexity = complexities[0]
                best_structure  = structures[0]
        

        if top_loss_MSE_history[-1] <= TOL:
            if print_op == True:
                print('Function with loss < TOL found')
            break

        # elif (i >= 10) & (False not in (torch.tensor(top_loss_nofreq_history[-10:-1]) == torch.ones_like(torch.tensor(top_loss_nofreq_history[-10:-1]))*top_loss_nofreq_history[-1])):
        #     print('Loss not changed in last 10 iterations')
        #     break


        elif (i >= 14):
            if top_loss_MSE_history[-15] <= best_loss:
                if print_op == True:
                    print('Loss not improved in last 15 iterations')
                break
            else:
                continue

    return best_sol, best_loss, best_frequency, best_complexity, best_structure, token_seqs_padded, losses, top_loss_history, top_loss_MSE_history, top_frequency_history, top_complexity_history, top_structure_history,  all_loss_MSE_history, all_frequency_history, all_complexity_history, all_structure_history,  i+1





def evolution_model_space(token_seqs_padded, data, binary_operators, unary_operators, leaves, GPU_device,  population_size = 40, iterations=100, TOL = 0.1, alpha = 0, beta = 0.1, gamma = 0, mut_prob = 0.6, print_op = False):
    """
    Evolves the population using mutation and crossover operators (for now just mutation). 
    Discards unfit members of the population and maintains constant population size (by discarding remaining members)
    Evloves without parallelisation. 
    Input: 
        token_seqs_padded: (torch.tensor) tensor containing all token seqences to be evolved (initial population)
        mapping to torch operations rather than sympy purely for evaluation.
                                also contains the mappings of calculated subtrees.
        data: (torch.tensor) (n,2) shaped array of experimanetal data
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
        GPU_device: (str) 'cuda' or 'mps' for GPU accelleration
        population_size: (int) size of population
        number_to_mutate: (int) how many tokens to mutate
        iterations: (int) number of evolution iterations to do
        threshold: (float) the threshold for extreme values (should be set as some (safety factor)*max(abs(data)))
    Outputs
        new_population_tokens: (np.array) array of evolved expressions (in token form)

    """
    top_loss_history = []
    top_loss_MSE_history = []

    top_frequency_history   = []
    top_complexity_history  = []
    top_structure_history   = []

    # Tracking Metrics of all solutions through evolution
    all_loss_MSE_history = []
    all_frequency_history   = []
    all_complexity_history  = []
    all_structure_history   = []

    binary_tokens = torch.arange(len(binary_operators), device=GPU_device)
    unary_tokens  = torch.arange(len(binary_operators), len(binary_operators)+len(unary_operators), device=GPU_device)
    leaf_tokens   = torch.arange(len(binary_operators)+len(unary_operators), len(binary_operators)+len(unary_operators)+len(leaves), device=GPU_device)
    None_token    = torch.arange(len(binary_operators)+len(unary_operators)+len(leaves), len(binary_operators)+len(unary_operators)+len(leaves)+1, device=GPU_device)
    # pad_token     = torch.tensor([-1])
    # tree_token    = torch.tensor([-2])          # Tree token for finding trees that arent complexity 3

    token_seqs_padded   = token_seqs_padded.to(GPU_device)
    data                = data.to(GPU_device)

    for i in tqdm(range(iterations), desc="Evolving", disable = not print_op):
        mutate_or_crossover = torch.rand(1, device=GPU_device)

        if mutate_or_crossover <= mut_prob:
            # Mutate and add mutated tokens to general population.
            modified_token_seqs = Search.mutate_operator(torch.clone(token_seqs_padded), binary_tokens, unary_tokens, leaf_tokens, None_token,GPU_device)

        else:
            which_pair = torch.randperm(token_seqs_padded.shape[0], device=GPU_device).reshape(int(token_seqs_padded.shape[0]/2),2)

            group1 = token_seqs_padded[which_pair[:,0]]
            group2 = token_seqs_padded[which_pair[:,1]]

            subtrees_start_idxs1 = [Search.identify_subtrees_old2(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token) for token_seq in torch.clone(group1)]
            subtrees_start_idxs1 = torch.nn.utils.rnn.pad_sequence(subtrees_start_idxs1, batch_first=True, padding_value=-1)

            subtrees_start_idxs2 = [Search.identify_subtrees_old2(token_seq, binary_tokens, unary_tokens, leaf_tokens, None_token) for token_seq in torch.clone(group2)]
            subtrees_start_idxs2 = torch.nn.utils.rnn.pad_sequence(subtrees_start_idxs2, batch_first=True, padding_value=-1)            

            modified_token_seqs_pre = Search.batched_crossover(torch.clone(group1), torch.clone(group2), torch.clone(subtrees_start_idxs1), torch.clone(subtrees_start_idxs2), GPU_device)
            modified_token_seqs     = torch.cat([modified_token_seqs_pre[:,0,:],modified_token_seqs_pre[:,1,:]])

        # candidate_token_seqs = pad_groups(token_seqs_padded, modified_token_seqs)
    
        # Evaluate fitness
        candidate_token_seqs = torch.cat([token_seqs_padded, modified_token_seqs])

        token_seqs_padded, losses, losses_MSE,  frequencies, complexities, structures = get_fittest_model_space(candidate_token_seqs, data, 
                                                                                                binary_tokens, unary_tokens, leaf_tokens, None_token, GPU_device,
                                                                                                alpha = alpha, beta = beta, gamma = gamma, number_of_sols = population_size)
        # if GPU_device != 'cpu':
        #     token_seqs_padded = token_seqs_padded_GPU.to('cpu')
        # else:
        #     token_seqs_padded = token_seqs_padded_GPU

        # if (i >= 2):
        #     if (top_loss_nofreq_history[-2] - top_loss_nofreq_history[-1]):
        #         print('Best Loss found in local search space')
        #         return token_seqs_padded, losses, top_loss_history, top_loss_nofreq_history, i

        top_loss_history.append(losses[0])
        top_loss_MSE_history.append(losses_MSE[0])
        top_frequency_history.append(frequencies[0].to('cpu'))
        top_complexity_history.append(complexities[0].to('cpu'))
        top_structure_history.append(structures[0].to('cpu'))

        all_loss_MSE_history.append(losses_MSE)
        all_frequency_history.append(frequencies.to('cpu'))
        all_complexity_history.append(complexities.to('cpu'))
        all_structure_history.append(structures.to('cpu'))

        if i == 0:
            best_sol  = token_seqs_padded[0]
            best_loss = losses_MSE[0]        # MSE losses

            best_frequency  = frequencies[0]
            best_complexity = complexities[0]
            best_structure  = structures[0]

        elif i >= 1:
            if top_loss_MSE_history[-1] < best_loss:
                best_sol  = token_seqs_padded[0]
                best_loss = losses_MSE[0]

                best_frequency  = frequencies[0]
                best_complexity = complexities[0]
                best_structure  = structures[0]

        if top_loss_MSE_history[-1] <= TOL:
            if print_op == True:
                print('Function with loss < TOL found')
            # return token_seqs_padded, losses, top_loss_history, top_loss_nofreq_history, i+1
            break

        # elif (i >= 10) & (False not in (torch.tensor(top_loss_nofreq_history[-10:-1]) == torch.ones_like(torch.tensor(top_loss_nofreq_history[-10:-1]))*top_loss_nofreq_history[-1])):
        #     print('Loss not changed in last 10 iterations')
        #     # return token_seqs_padded, losses, top_loss_history, top_loss_nofreq_history, i+1
        #     break

        elif (i >= 14):
            if top_loss_MSE_history[-15] <= best_loss:
                if print_op == True:
                    print('Loss not improved in last 15 iterations')
                break
            else:
                continue
        
    return best_sol, best_loss, best_frequency, best_complexity, best_structure, token_seqs_padded, losses, top_loss_history, top_loss_MSE_history, top_frequency_history, top_complexity_history, top_structure_history,  all_loss_MSE_history, all_frequency_history, all_complexity_history, all_structure_history,  i+1

