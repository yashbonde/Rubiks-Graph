'''
basic_generation.py

This file has a simple experiment for generation of graphs. The aim of this script is to
get structred graph knwoledge from any given input text. For now the input texts are
sample texts from Wikipedia. 

07.10.2019 - @yashbonde
'''

# import dep
import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp # used for sampling methods
# import graph_nets as gn # RubiksFrame is heavily inspired from this library

# somethings need to be custom
from common import shapes_list, positions_for, dense

# constants
TEXT_DIM = 8 # text embedding dim
NODE_DIM = 5 # embedding dimension for each node --> this will be expanded as the neural network
    # does more things from the information stored in each node
EDGE_DIM = 3 # embedding dimension for each edge
NUM_NODES_TYPES = 12 # say there are NUM_NODES_TYPES different types of nodes, which hold information
STOP_NODE = NUM_NODES_TYPES + 1 # this is the node which means no longer to add nodes to the graph
NUM_EDGES_TYPES = 9 # say there are 9 different types fo edges that hold information
MAX_EDGES = 30 # say that there are maximum 40 connections of knowledge points for the input
    # wiki information
MAX_NODES = 20 # say there can be max 20 different node points (things we want to extract from
    # knowledge)
MAX_ITER = 30 # this is the maximum number of iteration we allow for graph gen loop
MAXLEN_SEQ = 60 # we are capping this to 60 tokens
START_NODE_IDX = 0 # this is the start of the sequence of graph generation

# keys used in namedtuple
PH_KEYS = ['senders', 'receivers', 'edge_types', 'sender_types', 'receiver_types']
ALL_KEYS = PH_KEYS + ['node_embeddings', 'edge_embeddings']

"""Child Class of namedtuple need to be used as values cannot be changed once built"""
class Rubiks(namedtuple('Rubiks', ','.join(ALL_KEYS))):
    @classmethod
    def from_dict(cls, ext_dict):
        for k in ALL_KEYS:
            assert k in ext_dict, f"key: {k} not found in input dictionary"
        ordered_list  = [ext_dict[k] for k in ALL_KEYS]
        return cls(*ordered_list)

    @classmethod
    def from_list(cls, ext_list):
        # assuming the input is going to be in same order ad ALL_KEYS
        return cls(*ext_list)

"""functions"""
def get_simple_graph():
    """
    Before making the graph I am explainig how the graphs will be structired. I think that edges
    are going to be more important than the nodes to represent the graph information. This means
    that the graph is centred around many different types of edges.

    Thus the graph is a giant list of following tuple (sender_node, receiver_node, edge_type).
    """
    _num_edges = np.random.randint(low = int(MAX_EDGES/ 1.5), high = MAX_EDGES)
    # we define the sender and receiver index (order)
    senders = np.random.randint(MAX_NODES, size = (_num_edges,))
    receivers = np.random.randint(MAX_NODES, size = (_num_edges,))
    # we define the types --> used to get the emebeddings for nodes and edges
    edges_types = np.random.randint(NUM_EDGES_TYPES, size = (_num_edges,))
    senders_types = np.random.randint(NUM_NODES_TYPES, size = (_num_edges,))
    receivers_types = np.random.randint(NUM_NODES_TYPES, size = (_num_edges))
    return {
        "senders": senders,
        "receivers": receivers,
        "edge_types": edges_types,
        "sender_types": senders_types,
        "receiver_types": receivers_types
    }

def process_sequence(lines):
    """take lines as fetures and return vocab"""
    line = '\n'.join(lines)
    line = line.lower()
    vocab = list(set(line.split()))
    vocab = {c:i for i,c in enumerate(vocab)}
    lines = line.split('\n')
    return lines, vocab

"""
The three functions below deal with morphing of graphs either as targets or
as placeholder values. Diffculty that I have is that how are we going to get
differentiation between the different graphs. 

There are multiple ways to do this:
1. Using a graph_idx tensor which stores values of of which nodes belong
    to which graph.
    graph_idx = [0, 0, 0, 1, 1]
    receivers = [1, 2, 3, 2, 4]

2. Another way to use this is to force the graph generation till a prticular
    length and maskout the values that are after one graph in the batch is
    done. This is something similar to what is done in beam searching in NMT
    tasks.

For now I am preferring the second version over the first as there is less code
to be written for it.

In order to make the graph ready as a target func `get_target_from_graph_list`
is used whereas to make the placeholder from it we use function
`get_placeholders_from_graph_lists`.

Each graph we obtain will have the structure as follows:
- senders: [bs, num_graph,] int
- receivers: [bs, num_graph,] int
...

The second dimension will have to be same for tensorflow to process the
target graph. So we stack the values with 0 to make tensor a consistent
shape, or use `tf.RaggedTensor` (very tricy to use, maybe use `tf.nest` methods)
"""

def get_target_from_graph_list(graphs):
    max_len = max([len(g["senders"]) for g in graphs])
    empty = np.array([
        [np.zeros(max_len).astype(np.int32) for _ in range(len(PH_KEYS))]
            for _ in range(len(graphs))])
    for gidx, graph in enumerate(graphs):
        for kidx, key in enumerate(PH_KEYS):
            empty[gidx][kidx][:len(graph[key])] = graph[key]
    return empty

def get_placeholders_from_target():
    """take the target graphs and create placeholders from it"""
    rfph = namedtuple('RubiksFrame', PH_KEYS)
    for key in PH_KEYS:
        tensor = tf.placeholder(tf.int32, [None, None,], name = 'ph_{}'.format(key))
        setattr(rfph, key, tensor)
    return rfph

"""
this set of functions deal with the input sequence
>> simple attention method to force dynamic sequence length to a fixed size tensor
"""
def attention_reduce(tensor):
    outdim = shapes_list(tensor)[-1]
    # simple attention method to get the context vector
    # https://www.tensorflow.org/tutorials/text/nmt_with_attention
    att_k = tf.get_variable('attention_weight_key', [outdim, outdim])
    att_q = tf.get_variable('attention_weight_query', [outdim, outdim])
    att_v = tf.get_variable('attention_weight_wrapper', [outdim, 1])
    scores = tf.matmul(tf.nn.tanh(
        tf.matmul(tensor, att_k) + tf.matmul(tensor, att_q)
    ), att_v) # [bs, seqlen, 1]
    scores = tf.nn.softmax(scores, axis = 1) # [bs, seqlen, outdim]
    context_vector = tf.reduce_sum(tensor * scores, axis = [1]) # [bs, 1, outdim]
    # return tf.squeeze(out, axis = [1])
    return context_vector # [bs, outdim]

def text_feature_extractor(inp_seq, vocab_size, embedding_dim, outdim, maxlen = MAXLEN_SEQ):
    """simple feature extractor as a placeholder fo more complex architectures
    input: [batch_size, sequence_len]
    output: [batch_size, out_dim]"""
    with tf.variable_scope('embedding'):
        wte = tf.get_variable('text_matrix',
                            [vocab_size, embedding_dim],
                            initializer = tf.random_normal_initializer(
                                stddev = 0.02
                            ))
        wpe = tf.get_variable('position_matrix',
                            [maxlen, embedding_dim],
                            initializer = tf.random_normal_initializer(
                                stddev = 0.01
                            ))

        out = tf.gather(wte, inp_seq) + tf.gather(wpe, positions_for(inp_seq, 0)) # [bs, seqlen, edim]
        for layer_idx in range(10):
            out = dense(out, 'layer_{}'.format(layer_idx), embedding_dim)
        out = dense(out, 'final_projection', outdim)  # [bs, seqlen, outdim]
        return attention_reduce(out)
     
"""next set of function deal with the network generating graph"""
def concatenate_node_edge_tensors(rf):
    # rf --> [bs, num_edges]
    # gather --> [bs, num_edges, edim]
    sender_nodes_info = tf.gather(rf.node_embeddings, rf.senders) + tf.gather(rf.node_embeddings, rf.sender_types)
    rece_nodes_info = tf.gather(rf.node_embeddings, rf.receivers) + tf.gather(rf.node_embeddings, rf.receiver_types)
    edge_info = tf.gather(rf.edge_embeddings, rf.edge_types)
    return tf.concat([sender_nodes_info, rece_nodes_info, edge_info], axis = -1) # combine at the last dimension

def get_graph_embedding(rf, hidden_dim):
    with tf.variable_scope("graph_emebedding"):
        out = concatenate_node_edge_tensors(rf) # [bs, num_edges, d1 + d2 + d3]
        for layer_idx in range(10):
            out = dense(out, 'layer_{}'.format(layer_idx), hidden_dim)
        out = dense(out, 'final_projection', hidden_dim)  # [bs, seqlen, outdim]
        return attention_reduce(out) # reduce using attention # [bs, outdim]

def add_node(rf, text, hidden_dim, num_layers = 3, num_node_type = NUM_NODES_TYPES + 1):
    """add node function, returns a probability sampled node"""
    with tf.variable_scope('add_node'):
        graph_embedding = get_graph_embedding(rf, hidden_dim) # get graph embedding
        out = tf.concat([graph_embedding, text], axis = -1) # join text and graph along last dim
        # join the information tensors
        for layer_idx in range(num_layers):
            out = dense(out, 'layer_{}'.format(layer_idx), hidden_dim) # [bs, hidden_dim]
        out = tf.nn.softmax(dense(out, 'projection', num_node_type)) # [bs, num_node_types]
        out = tf.random.categorical(out, 1) # [bs, 1]
    return out

def add_edge(rf, new_node, text, hidden_dim):
    with tf.variable_scope('add_edge'):
        graph_embedding = get_graph_embedding(rf, hidden_dim) # [bs, outdim]
        new_node_info = tf.gather(rf.node_embeddings, new_node) # [bs, 1, node_dim]
        new_node = tf.squeeze(new_node_info, axis = [1]) # [bs, node_dim]
        # join the information tensors
        out = tf.concat([graph_embedding, new_node, text], axis = -1) # [bs, d1 + d2 + d3]
        for layer_idx in range(10):
            out = dense(out, 'layer_{}'.format(layer_idx), hidden_dim)
            
        '''
        # this is the approach we use when going for the paper method
        out = tf.nn.sigmoid(dense(out, 'final_projection', 1))  # sigmoid()[bs, 1]
        bernoulli = tfp.distributions.Bernoulli(probs = out)
        return bernoulli.sample()
        '''
        
        # for now will progress with simple categorical method
        out = dense(out, 'projection', NUM_EDGES_TYPES)
        out = tf.random.categorical(out, 1) # [bs, 1]
    return out

def add_nodes(rf, new_node, text, new_edge, hidden_dim, max_pos_nodes = MAX_ITER):
    with tf.variable_scope('add_nodes'):
        graph_embedding = get_graph_embedding(rf, hidden_dim) # [bs, outdim]
        new_node = tf.squeeze(tf.gather(rf.node_embeddings, new_node), axis = [1]) # [bs, node_dim]
        new_edge = tf.squeeze(tf.gather(rf.edge_embeddings, new_edge), axis = [1]) # [bs, edge_dim]
        # join the information tensors
        out = tf.concat([graph_embedding, new_node, new_edge, text], axis = -1) # [bs, d1 + d2 + d3 + d4]
        for layer_idx in range(10):
            out = dense(out, 'layer_{}'.format(layer_idx), hidden_dim)
            
        """NOTE: this is the super tricky part is completely dependent on the GLOBAL values
        since in this step it has to return which node is it going to connect to and thus
        the projection is done to the maximum number of nodes in the experimental setting.
        
        We then slice it to the number of nodes currently available to connect to. We then
        perform softmax over this slice to only get the selection from the number of nodes
        in the graph till now.
        
        There is another method we can do this, it can be masking the value which are not
        possible by zeros and then performing softmax over those. The two reasons it is better
        than the approach above:
        1. It gives the same results as the above version
        2. It is more tensorflow-like wherein splitting etc. can cause issues
        """
        out = dense(out, 'node_projection', max_pos_nodes) # [bs, MAX POSSIBLE NODES]
        
        with tf.name_scope("masking"):
            # first we make a mask using the point will which we have the nodes till now
            nodes_padded = tf.pad(rf.senders, [[0, 0], [0, max_pos_nodes+1-tf.shape(rf.senders)[0]]], constant_values = int(-1))
            mask = tf.cast(tf.where(
                nodes_padded > tf.ones_like(nodes_padded) * -1,
                out,
                tf.ones_like(out) * int(-1e10)
            ), out.dtype)[:, :tf.shape(rf.senders)[0]-1]
            masked_softmax = tf.nn.softmax(mask)
            
            # now we only have to return a single value, the best approach is to do tf.nn.top_k, unlike tf.argmax
            # tf.math.top_k is differentiable and thus can be used to flow gradients
            out = tf.math.top_k(masked_softmax, k = 1).indices # [bs, 1]
    return out

def add_receiver_node(rf, new_node, text, hidden_dim, num_layers = 3, num_node_type = NUM_NODES_TYPES + 1):
    """get the receiver node type, returns a probability sampled node"""
    with tf.variable_scope('receiver_node_type'):
        graph_embedding = get_graph_embedding(rf, hidden_dim) # get graph embedding
        new_node_info = tf.gather(rf.node_embeddings, new_node) # [bs, 1, node_dim]
        new_node = tf.squeeze(new_node_info, axis = [1]) # [bs, node_dim]
        out = tf.concat([graph_embedding, new_node, text], axis = -1) # join text and graph along last dim
        # join the information tensors
        for layer_idx in range(num_layers):
            out = dense(out, 'layer_{}'.format(layer_idx), hidden_dim) # [bs, hidden_dim]
        out = tf.nn.softmax(dense(out, 'projection', num_node_type)) # [bs, num_node_types]
        out = tf.random.categorical(out, 1) # [bs, 1]
    return out

"""next function is the main maker function"""
def make_graph_from_text(text, rf, max_iter):
    """
    This function takes in the text and genrates a graph according to it. Since the
    process of graph generation requires looping, we use the `tf.while_loop` as
    iterator. `tf.while_loop` uses a body and condition function and the inputs
    and outputs must be of the same shape and count.
    :param text: text embedding [bs, graph_dim]
    
    NOTE:
    using the graph generation paper from deepmind as the baseline method for generating
    the graphs we only change the network architectures. The method has three steps as
    follows:
    1. to propogate over the complete network and get the graph embesdding [use MLP]
    2. use this graph embedding to decide whether to add a node or not, depends
        upon categorical distribution, one of the nodes is STOP node [use MLP]
        $p_t^{addnode} = f_{addnode}(G_{t-1})$
    3. next the node is added and for each node other than the recently added
        we caculate the probability to add the edge by going over all the nodes
        till now. This gives a Bernoilli distribution and we sample according to
        that.
        $p_{t,i}^{addedge} = f_{addedge}((V_t, E_{t,0}), v_t)$
    4. next up we decide the node we want to connect to
        $p_{t,i}^{nodes} = f_{nodes}((V_t, E_{t,0}), v_t)$
            
    The challenge here is two fold how can I combine the above method with the seq2seq
    kind of beam decoding that I want to do. The authors used a very comple LSTM network
    kind of thing to generate the graph. Using that is way out of scope for this quick
    hack. What are the information points that I have:
        1. embedding of sequence of shape [None, text_emb_dim]
        2. graphs with shapes [bs, num_edge,]
        3. node and edge embedding of shape [num_types, edim]
        
    Using this we need to create a graph embedding which can be done using simple MLP.
    
    I have found one issue with the current approach, which is it cannot generate 
    """
    batch_size, text_emb_dim = shapes_list(text)
    hidden_dim = text_emb_dim * 4 # using the default from transformer network
    
    def step(loop_idx, text, graph = None):
        # take one step --> called from body function
        if graph is None and text is None:
            raise ValueError("need to provide either text and/or graph")

        elif graph is None:
            """
            this is the case for the first step. In this particular section we need
            to create the graph from text embeddings and from the next iteration we will
            use the graph and text combined to get the graph 
            """
            graph = namedtuple('rfgen', PH_KEYS)
            for key in PH_KEYS:
                setattr(graph, key,
                    tf.ones_like(getattr(rf, key), name = 'gen_{}'.format(key)) * START_NODE_IDX
                )
            return text, graph
        
        else:
            assert text is not None
            # this is where the magic happens, we have the yet generated graph and text emebddigns
            # using this we predict the next tuple of [sender, receiver, graph_type]
            
            # step 1: decide the type of node to add
            node_type_to_add = add_node(rf, text, hidden_dim)
            
            """ideally what we need to add here is the condional that if the category is
            STOP then exit the loop for generation, but this is a different take on this
            approach thus we cannot do this.
            if node_type_to_add == STOP_NODE:
                exit generation loop 
            """
            sender_node = tf.cond(
                node_type_to_add == STOP_NODE,
                node_type_to_add,
                tf.random.uniform([batch_size, 1], maxval = STOP_NODE - 1, dtype = node_type_to_add.dtype)
            ) # [bs, 1]
            
            node_idx = tf.identity(loop_idx) # what is the idx of this current node this is equal to the loop idx
            
            # step 2: decide if we want to add an edge
            edge_types = add_edge(rf, sender_node, text, hidden_dim)
            
            """ideally here we would add a conditional, that only if return is add edge we continue further
            and determine the node with which to add edge
            """
            
            # step 3: decide which node to add from the yet generated nodes
            reciever_node = add_nodes(rf, sender_node, text, edge_types, hidden_dim) # [bs, 1]
            reciever_node = tf.gather_nd(tf.expand_dims(
                tf.concat(
                    [tf.expand_dims(tf.range(tf.shape(reciever_node)[0]), axis = 1), reciever_node],
                    axis = 1
                )
            ), axis = 1) # [bs, 1]

            # step 4: decide on the type of receiver node
            reciever_node_type = add_receiver_node(rf, reciever_node, text, hidden_dim)
            
            # step 5: add this to our graph tuple
            rf.senders = tf.concat([rf.senders, node_idx], axis = 1)
            rf.receivers = tf.concat([rf.recievers, reciever_node], axis = 1)
            rf.edges = tf.concat([rf.edges, edge_types], axis = 1)
            rf.sender_types = tf.concat([rf.sender_types, sender_node])
            rf.reciever_types = tf.concat([rf.reciever_types, reciever_node_type], axis = 1)
            # ---> done with the step --->

            return rf

    with tf.name_scope("graph_gen"):
        # we first take a step to get the initial tensor and then use is as in input
        # to the bod

        graph = step(text)

        def body(text, rf):
            """shapes for each of the tensors is given in the functions above"""
            return 

        def cond(_):
            return True
        
        # define the shape invariants
        shape_invariants = [
            tf.TensorShape([None, text_emb_dim]),
            tf.TensorShape([None,]),
            tf.TensorShape([None,]),
            tf.TensorShape([None,])
        ]  # TODO: Finalise the shapes and add here

        _, senders, receivers, edge_types, sender_types, receiver_types = tf.while_loop(
            body,
            cond,
            loop_vars = [text, rf],
            back_prop = True,
            maximum_iterations = max_iter,
            shape_invariants = shape_invariants
        )

        return None


if __name__ == "__main__":
    """
    NOTE: couple of things before you start understanding the code. Batch size in this example
          are the number of different graphs that we use for training. There can be a lot of
          confusion in understanding the dimensions of the model.
    """
    # first thing is to load the texts and remove the commented lines
    lines = [line for line in open('./wiki_sample.txt').readlines() if line[0] != '#']
    lines, vocab = process_sequence(lines)

    # now make graphs, target graphs and placeholders from those target graph
    graphs = [get_simple_graph() for _ in range(len(lines))]
    target_graphs = get_target_from_graph_list(graphs)
    rf = get_placeholders_from_target()
    setattr(rf, 'node_embeddings', tf.get_variable(name = 'node_emb',
                                        shape = [NUM_NODES_TYPES, NODE_DIM],
                                        initializer = tf.random_normal_initializer(
                                            stddev = 0.02
                                        ))) # further setting value for node embedding matrix
    setattr(rf, 'edge_embeddings', tf.get_variable(name = 'edge_emb',
                                        shape = [NUM_EDGES_TYPES, EDGE_DIM],
                                        initializer = tf.random_normal_initializer(
                                            stddev = 0.02
                                        ))) # further setting value for edge embedding matrix

    """
    Now we make placeholder for the input sentence. Note that this structure depends
    on the kind of network we are going to use for extracting features from sentences.
    For now assuming the simple transformer network as text feature extractor.
    """
    text_placeholder = tf.placeholder(tf.int32, [None, None], name = 'text_placeholder')
    text_features =  text_feature_extractor(text_placeholder, len(vocab), TEXT_DIM, int(TEXT_DIM * 3))

    for key in ALL_KEYS:
        print(getattr(rf, key))
    print(text_features)

    # next up we feed the text features and RubiksFrame to the graph gen model
    graph = make_graph_from_text(text_features, rf, MAX_ITER)