
# # Generation of Random Functions Script

# Modified code from https://github.com/benmoseley/symbolic-regression-with-deep-neural-networks-workshop by Dr Ben Moseley



import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from IPython.display import display
import json
import gzip

from tqdm import tqdm
import warnings
from numpy.lib.stride_tricks import sliding_window_view

import Evaluation



# Define library of binary operators
binary_operators = {
    "+": lambda l, r: l + r,
    "-": lambda l, r: l - r,
    "*": lambda l, r: l * r,
    "/": lambda l, r: l / r,
}

# Define library of unary operators
unary_operators = {
    # "abs": lambda o: sp.Abs(o),
    "sin": lambda o: sp.sin(o),
    # "tan": lambda o: sp.tan(o),
    "exp": lambda o: sp.exp(o),
    # "^-1": lambda o: sp.Pow(o,-1),
}

leaves = ("x")



# Define procedure for generating random expression in the form of trees
class TreeNode:
    """A node in a binary tree representing a mathematical expression.

    Attributes:
        val (str): The operator value at this node.
        left (TreeNode or str): The left child (subtree or leaf).
        right (TreeNode or str): The right child (subtree or leaf).
    """
    def __init__(self, val=None, left=None, right=None):
        self.val = val      # Node value: operator
        self.left = left    # Left child node or leaf
        self.right = right  # Right child node or leaf

def generate_random_binary_tree(n_operators, binary_operators, leaves):
    """Generates a random expression tree consisting of binary operators.

    Args:
        n_operators (int): Number of binary operators to include in the tree.
        binary_operators (dict): Mapping from binary symbols to binary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).

    Returns:
        TreeNode: The root of the randomly generated expression tree.
    """
    # Create a new root node with a random operator
    root = TreeNode()
    root.val = random.choice(list(binary_operators.keys()))
    n_operators -= 1

    # Randomly split the remaining operators between left and right subtrees
    left_n_operators = random.randint(0, n_operators)
    right_n_operators = n_operators - left_n_operators

    ## TODO: assign `root.left` and `root.right` attributes
    if left_n_operators > 0:
        root.left = generate_random_binary_tree(left_n_operators,binary_operators,leaves)
    else:
        root.left = random.choice(list(leaves))

    if right_n_operators > 0:
        root.right = generate_random_binary_tree(right_n_operators,binary_operators,leaves)
    else:
        root.right= random.choice(list(leaves))
    ##
    return root


def str_tree(root, level=0, prefix="Root "):
    """Returns a pretty string representation of a TreeNode expression tree.

    Args:
        root (TreeNode or str): The current node or leaf.
        level (int): The current depth in the tree (for indentation).
        prefix (str): Prefix label for the node (e.g. 'Root ', 'L-- ').

    Returns:
        str: A formatted string representing the tree.
    """
    indent = " " * (level * 4)
    if isinstance(root, TreeNode):
        s = indent + prefix + str(root.val) + "\n"
        s += str_tree(root.left, level + 1, "L-- ")
        s += str_tree(root.right, level + 1, "R-- ")
        return s
    else:  # Leaf node
        return indent + prefix + str(root) + "\n"
    

def add_random_unary_operators(root, n_operators, unary_operators):
    """Adds unary operations randomly to internal nodes of a binary expression tree.

    Args:
        root (TreeNode): The root of the binary expression tree.
        n_operators (int): Number of unary operations to insert.
        unary_operators (dict): Mapping from unary symbols to unary functions.

    Returns:
        TreeNode: The root of the updated expression tree.
    """
    
    if not isinstance(root, TreeNode):
        raise Exception(f"root is not a TreeNode, {root}")
    
    if n_operators == 0:
        return root

    # Collect all internal (non-leaf) nodes into a flat list
    nodes = []
    def collect_nodes(node):
        if isinstance(node, TreeNode):
            nodes.append(node)
            collect_nodes(node.left)
            collect_nodes(node.right)
    collect_nodes(root)

    # Randomly add unary operations
    for _ in range(n_operators):

        # Randomly choose a node to wrap in a unary operator
        chosen_node = random.choice(nodes)

        ## TODO: wrap the chosen in a new TreeNode, `new_node`.
        ## The new node should have a unary value, a left attribute as the chosen node, and `None` as the right attribute
        new_node = TreeNode()

        new_node.val = random.choice(list(unary_operators.keys()))
        new_node.left = chosen_node
        new_node.right = None
        ##

        # Update references to the chosen node in the tree
        if root is chosen_node:
            root = new_node
        else:
            for node in nodes:  # Update in-place references
                if node.left is chosen_node:
                    node.left = new_node
                elif node.right is chosen_node:
                    node.right = new_node

        # Add the new unary node to the list for possible further wrapping
        nodes.append(new_node)
    return root


def print_tree(root):
    """Pretty-prints the tree structure."""
    print(str_tree(root))


# Function for coverting between tree structure (for manipulation) and sympy (for reading and calculations)
def tree_to_sympy(root, binary_operators, unary_operators, leaves):
    """Converts a binary tree expression into a SymPy expression.

    Args:
        root (TreeNode): The root of the binary expression tree.
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).

    Returns:
        sympy.Expr: The SymPy expression corresponding to the tree.
    """
    if isinstance(root, TreeNode):
        if root.val in binary_operators:


            ## TODO: complete building the sympy expression for a binary operator.
            ## Recursively build left and right `sympy` expressions, then evaluate
            ## and return the binary `sympy` expression.
            left_expr  =    tree_to_sympy(root.left, binary_operators,unary_operators,leaves)
            right_expr =    tree_to_sympy(root.right,binary_operators,unary_operators,leaves)

            return binary_operators[root.val](left_expr,right_expr)
            ##

        elif root.val in unary_operators:


            ## TODO: complete building the sympy expression for a unary operator.
            ## Recursively build left `sympy` expression, then evaluate and return
            ## the unary `sympy` expression.
            left_expr = tree_to_sympy(root.left,binary_operators,unary_operators, leaves)
            return unary_operators[root.val](left_expr)
            ##

        else:
            raise Exception(f"root value not recognised: {root.val}")  # Unrecognized operator
    else:  # leaf node
        assert root in leaves  # Ensure the root is a valid leaf value
        expr = sp.Symbol(root)  # Convert leaf value to a SymPy symbol
        return expr
    

# Generation of Population of random expressions

# General changes:
# * Omitting the migration step in PySR that utilizes parallelization (for now). Therefore only one population will be generated.
# * No hard constraint on the total complexity of a generated expression. Constraints on max number of unary and binary operators imposed instead

def generate_expr_population(max_n_binary_operators,max_n_unary_operators,expression_number, variable_values,threshold,  binary_operators,unary_operators,leaves, input_seed = 123):
    """
    Function Generates expressions up to a given number of binary operators and unary operators. 
    Expressions are filtered to remove extreme values (avoid over errors) and duplicate expressions.
    Inputs:
        max_n_binary_operators:  (int) Maximum number of binary operators
        max_n_unary_operators: (int) Maximum number of unary operators
        expression_number: (int) number of expressions to generate
        variable_values: (np.array) array of values for each variable (100 for x, 100 for y)
        x_values: (np.array) numpy array containing all the x_values for calculation of expression value.
        threshold: (float) +ve number that manages the threshold for an extreme expression

        binary_operators: (dict) Library of binary operators. (includes lambda function in value part for computation)
        unary_operators: (dict) Library of unary operators. (includes lambda function in value part for computation)
        leaves: (tuple) Library of variables used in the generated expressions. For current iter should only contain one variable (x).
    Outputs:
        roots_filtered: (np.array) list of root nodes of the generated expressions in tree form 
        exprs_filtered: (np.array) list of expressions using sympy pretty printing (corresponds to roots output)
        datas_filtered: (np.array) input and corresponding outputs for an expression
    """

    # Lists to store the generated tree roots and corresponding SymPy expressions
    roots = []
    exprs = []

    random.seed(input_seed)
    for i in tqdm(range(expression_number),desc = "Generating Trees"):

        # Randomly sample the size of the tree
        n_binary_operators = random.randint(1, max_n_binary_operators)
        n_unary_operators = random.randint(0, max_n_unary_operators)


        ## TODO: generate a random tree and append it to `roots`.
        ## Also convert the tree to its `sympy` expression and append it to `exprs`.
        generated_tree = generate_random_binary_tree(n_binary_operators, binary_operators, leaves)
        generated_tree = add_random_unary_operators(generated_tree, n_unary_operators, unary_operators)
        generated_expr = tree_to_sympy(generated_tree,binary_operators,unary_operators,leaves)

        roots.append(generated_tree)
        exprs.append(generated_expr)
        ##
    print('Generation Complete')

    # List to store the observational data for each expression
    data_checks = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for expr in tqdm(exprs,desc = "Calculating Function Outputs"):

            ## TODO: Compute observational `data` for each expression and append it to `datas`.
            ## `data` should be a `numpy` array of shape (N, 2), where the first column contains
            ## `x_values`, and the second column contains `f(x_values)`.
            try:

                fn = sp.lambdify(args=sp.symbols(leaves), expr=expr, modules="numpy")
                f_values = fn(*variable_values)

                if isinstance(f_values, (int, float)):
                    # f_values = np.ones(len(variable_values.squeeze())) * f_values
                    f_values = np.ones(len(variable_values.squeeze())) * 0
                    data_check = np.stack([*variable_values, f_values], axis=-1)
    
                else:
                    data_check = np.stack([*variable_values, f_values], axis=-1)
                data_checks.append(data_check)  # Evaluate the function on the input values
                ##

            except:
                f_values = np.ones(len(variable_values.squeeze())) * 10e10

                if isinstance(f_values, (int, float)):
                    f_values = np.ones(len(variable_values.squeeze())) * 0
                    data_check = np.stack([*variable_values, f_values], axis=-1)
    
                else:
                    data_check = np.stack([*variable_values, f_values], axis=-1)
                data_checks.append(data_check)  # Evaluate the function on the input values
                ##


            # Ensure that the data is a numpy array with shape (100, 2)
            assert isinstance(data_check, np.ndarray) and data_checks[-1].shape == (len(variable_values[0]), len(variable_values)+1)

    print('Calculations Complete')
    # Lists to store the filtered roots, expressions, and corresponding data
    roots_filtered          = []
    exprs_filtered          = []
    data_checks_filtered    = []
    data_checks_unique      = []  # To keep track of unique tree structures (to filter duplicates)
    roots_unique            = []  # To keep track of unique tree structures (to filter duplicates) (older version maybe ok to delete)

    # Loop through all the generated trees, expressions, and their observational data
    for root, expr, data_check in tqdm(zip(roots, exprs, data_checks), desc= "Filtering Expressions"):

        ## TODO: Filter out extreme expressions and duplicate trees from `roots`, `exprs`, and `datas`.
        if (np.max(np.abs(data_check[:,1])) <= threshold) and (np.mean(np.abs(data_check[:,1])) != 0) and not any(np.array_equal(data_check[:,1], arr[:,1]) for arr in data_checks_unique) and (str_tree(root) not in roots_unique):
            roots_filtered.append(root)
            roots_unique.append(str_tree(root))
            data_checks_unique.append(data_check)
            exprs_filtered.append(expr)
            data_checks_filtered.append(data_check)
        ##


    # Print the number of trees remaining after filtering, and how many were removed
    print(f"Total number of expressions in filtered population: {len(roots_filtered)} ({len(roots)-len(roots_filtered)} expressions removed)")

    return np.array(roots_filtered), np.array(exprs_filtered), np.array(data_checks_filtered)


# Conversions between tokens, trees and expressions

# Tokenization to keep track of complexity + evolution operations

def create_token_maps(binary_operators, unary_operators, leaves):
    """
    Creates forward and backward token maps.

    Args:
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).

    Returns:
        forward_token_map (dict): Mapping from symbol to token index.
        backward_token_map (dict): Mapping from token index to symbol.
    """
    # Combine all tokens (binary operators, unary operators, leaves, special tokens)
    all_tokens = list(binary_operators.keys()) + list(unary_operators.keys()) + list(leaves) + [None,] + ["<EOS>"] + ["<PAD>"]

    # Create forward and backward token maps
    return ({token: idx for idx, token in enumerate(all_tokens)},
            {idx: token for idx, token in enumerate(all_tokens)})

def tree_to_prefix_sequence(root):
    """
    Converts a binary tree to a prefix sequence.

    Args:
        root (TreeNode): The root node of the tree.

    Returns:
        list: A list representing the prefix sequence of the tree.
    """
    if isinstance(root, TreeNode):
        return [root.val] + tree_to_prefix_sequence(root.left) + tree_to_prefix_sequence(root.right)
    else:  # Leaf node
        return [root]

def prefix_sequence_to_tree(sequence, binary_operators, unary_operators, leaves):
    """
    Converts a prefix sequence back to a binary tree.

    Args:
        sequence (list): The prefix sequence.
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).

    Returns:
        root (TreeNode): The reconstructed binary tree root.
    """
    def build_binary_tree(iterator):
        val = next(iterator)
        root = TreeNode()
        root.val = val

        # If the value is an operator, recursively build left and right subtrees
        if val in binary_operators or val in unary_operators:
            count_left, root.left = build_binary_tree(iterator)
            count_right, root.right = build_binary_tree(iterator)
            if val in unary_operators:
                assert root.right is None  # Unary operators should not have a right child
        else:
            # If it's a leaf node, return the value
            # assert val in leaves + (None,)  # Check consistency
            return 1, val  # Leaf node

        return 1 + count_left + count_right, root

    count, root = build_binary_tree(iter(sequence))
    # assert count == len(sequence)  # Ensure the number of nodes matches the sequence length (only use for generation initially)
    return root

# Create token maps 
forward_token_map, backward_token_map = create_token_maps(binary_operators, unary_operators, leaves)

# Functions to tokenise functions from trees
def tree_to_token(root):
    """
    Takes the root node from an expression and returns the corresponding token 
    Inputs: 
        root: (node) root node for the desired expression
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
    Output:
        token_seq: (torch.tensor) the corresponding token sequence
    """
    prefix_seq = tree_to_prefix_sequence(root)
    token_seq = torch.from_numpy(np.vectorize(lambda i: forward_token_map.get(i))(prefix_seq))
    return token_seq

def token_to_tree(token_seq, binary_operators, unary_operators,leaves):
    """
    Takes a token sequence from an expression and reconstructs the corresponding root node
    Inputs: 
        token_sequence: (torch.tensor) the corresponding token sequence
        binary_operators (dict): Mapping from binary symbols to binary functions.
        unary_operators (dict): Mapping from unary symbols to unary functions.
        leaves (tuple of str): Set of allowable leaf values (input variables).
    Output:
        root: (node) root node for the desired expression
    """
    prefix_seq = np.vectorize(lambda i: backward_token_map.get(i), otypes=[object])(token_seq).tolist()
    root = prefix_sequence_to_tree(prefix_seq, binary_operators, unary_operators,leaves)
    return root

def generate_candidate_token_sequences(population_size, max_n_binary_operators, max_n_unary_operators, max_complexity, data, binary_operators, unary_operators, leaves, input_seed=5):
    """
    Wraps the function generation and coversion to token sequences into one funtion. Returns padded token sequences as output
    
    """

    assert max_complexity % 2 !=0, "Max complexity must be odd to ensure valid token sequence generation"

    expression_number = 100*population_size
    variable_values = np.array([data[:,0].numpy()])
    threshold = 10*max(data[:,1]).float()

    tmp_roots, tmp_exprs, _ = generate_expr_population(max_n_binary_operators, max_n_unary_operators,
                                                       expression_number, variable_values, threshold,
                                                       binary_operators, unary_operators,leaves, 
                                                       input_seed)
    
    test_roots = tmp_roots[0:population_size]
    test_exprs = tmp_exprs[0:population_size]

    test_token_seqs = []
    for root in test_roots:
        test_token_seqs.append(tree_to_token(root))

    test_token_seqs_padded = torch.nn.utils.rnn.pad_sequence(test_token_seqs, batch_first=True, padding_value=-1)

    if test_token_seqs_padded.shape[1] < max_complexity:
        test_token_seqs_padded = torch.nn.functional.pad(test_token_seqs_padded, (0,max_complexity-test_token_seqs_padded.shape[1]),"constant", -1)

    assert test_token_seqs_padded.shape[1] == max_complexity, "Generated token sequences are more complex than the max complexity, try reducing max number of operators"

    return test_token_seqs_padded, test_roots, test_exprs

def generate_candidate_test_split_token_sequences(population_size, true_number, test_max_n_binary_operators, test_max_n_unary_operators, test_max_complexity, true_max_n_binary_operators, true_max_n_unary_operators, true_max_complexity, data, binary_operators, unary_operators, leaves, input_seed=5):
    """
    Wraps the function generation and coversion to token sequences into one funtion. Returns padded token sequences as output
    
    """

    assert (test_max_complexity % 2) and (true_max_complexity % 2 ) !=0, "Max complexity must be odd to ensure valid token sequence generation"

    variable_values = np.array([data.numpy()])
    threshold = 100*max(data).float()

    tmp_roots, tmp_exprs, outputs = generate_expr_population(test_max_n_binary_operators, test_max_n_unary_operators,
                                                       100*population_size, variable_values, threshold,
                                                       binary_operators, unary_operators,leaves, 
                                                       input_seed)
    test_token_seqs    = []

    for root in tmp_roots:
        test_token_seqs.append(tree_to_token(root))
    
    test_token_seqs_padded = torch.nn.utils.rnn.pad_sequence(test_token_seqs, batch_first=True, padding_value=-1)


    batched_complexity_eval = torch.vmap(Evaluation.complexity_eval)
    complexities = batched_complexity_eval(test_token_seqs_padded).squeeze()
    simplest_idxs =  torch.argsort(complexities)[:population_size]
    test_token_seqs_padded = test_token_seqs_padded[simplest_idxs]
    test_token_seqs_padded = test_token_seqs_padded.squeeze(dim=1)
    

    assert test_token_seqs_padded.shape[0] == population_size

    if test_token_seqs_padded.shape[1] < test_max_complexity:
        test_token_seqs_padded = torch.nn.functional.pad(test_token_seqs_padded, (0,test_max_complexity-test_token_seqs_padded.shape[1]),"constant", -1)

    assert test_token_seqs_padded.shape[1] == test_max_complexity, "Generated test token sequences are more complex than the test max complexity, try reducing max number of operators"

    print('-'*100)


    tmp_roots, tmp_exprs, outputs = generate_expr_population(true_max_n_binary_operators, true_max_n_unary_operators,
                                                       100*true_number, variable_values, threshold,
                                                       binary_operators, unary_operators,leaves, 
                                                       input_seed+1)
    

    true_roots  = tmp_roots[0:true_number]
    true_exprs  = tmp_exprs[0:true_number]
    true_data = torch.from_numpy(outputs[0:true_number])

    true_token_seqs    = []
    
    for root in true_roots:
        true_token_seqs.append(tree_to_token(root))

    true_token_seqs_padded = torch.nn.utils.rnn.pad_sequence(true_token_seqs, batch_first=True, padding_value=-1)

    if true_token_seqs_padded.shape[1] < true_max_complexity:
        true_token_seqs_padded = torch.nn.functional.pad(true_token_seqs_padded, (0,true_max_complexity-true_token_seqs_padded.shape[1]),"constant", -1)
    assert true_token_seqs_padded.shape[1] == true_max_complexity, "Generated true token sequences are more complex than the true max complexity, try reducing max number of operators"

    return test_token_seqs_padded, true_token_seqs_padded, true_roots, true_exprs, true_data


def Basic_differentiator(data):
    """
    Does a basic differentiation of the data by takin the difference between two points
    and divding by the step size.
    """
    step                = data[1,0] - data[0,0]
    f_rolled            = data[:,1].roll(1,dims=0)
    x_rolled            = data[:,0].roll(1,dims=0)
    differences         = (data[:,1]-f_rolled).roll(-1,dims=0) / step
    midpoints           = (data[:,0]+x_rolled).roll(-1,dims=0) / 2
    diff_data           = torch.stack([midpoints[:-1], differences[:-1]]).mT

    return diff_data