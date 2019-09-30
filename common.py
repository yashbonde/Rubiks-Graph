def base_transformer(inp, config, num_layers, scope = 'transformer'):
    with tf.variable_scope(scope):
        out = tf.expand_dims(inp, axis = 0)
        for layer_idx in range(num_layers):
            out = att_block(out, 'block_{}'.format(layer_idx), None, config)[0]
        return out[0]

def singular_direction_message(node_u, node_v, edges, gather_index_tensor, num_segmented_ids,
                               out_dim, scope = 'singular_direction_message'):
    """
    pass the message in a single direction

    :param node_u: all sender nodes tensors [num_edges, node_edim]; float
    :param node_v: all sender nodes tensors [num_edges, node_edim]; float
    :param edges: edge tensors [num_edges, edge_edim]; float
    :param gather_index_tensor: tensor with indices to gather [num_edge,]; int
    :param num_segmented_ids: tensor with 0 dim with number of nodes
    :param out_dim: output dimension from mlp
    :param scope: tf variable scope name

    :return: node aggregated tensor with shape [num_nodes, node_edim]
    """
    with tf.variable_scope(scope):
        concat = tf.concat([node_u, node_v, edges], axis = 1)
        mlp_out = mlp(concat, scope = 'm_fn', num_features = out_dim)
        
        """
        $\sum_{u:(u,v)\in E}{f_e(h_u, h_v, x_{u,v})}$
        where u: gather_index_tensor
        where h_u: node_u
        where h_v: node_v
        where x_{u_v}: edges
        """
        # unsorted_sum
        sum_ = tf.math.unsorted_segment_sum(
            mlp_out,
            gather_index_tensor,
            num_segmented_ids
        )
        
        return sum_
    
def message_pass(nodes, edges, senders, receivers, n_node, n_edge, out_dim, scope = 'message_passing'):
    """
    function to perform forward + reverse message passing
    
    :param nodes: nodes embedding tensor shape [num_node_types, node_edim]; float
    :param edges: edges feature tensor [num_edges, edge_edim]; float
    :param senders: tensor with sender indices [num_edge,]; int
    :param receivers: tensor with receiver indices [num_edge,]; int
    :param n_node: tensor with number of nodes by graph [num_graphs,]; int
    :param n_edge: tensor with number of edges by graph [num_graphs,]; int
    :param out_dim: output dimension of the message passing
    :param scope: tf variable scope name
    
    :return: aggregated message for number of nodes [num_nodes, out_dim]
    """
    with tf.variable_scope(scope):
        
        num_nodes = tf.reduce_sum(n_node)
        sender_data = tf.gather(nodes, senders)
        receiver_data = tf.gather(nodes, receivers)
        
        # \sum_{u:(u,v)\in E}{f_e(h_u, h_v, x_{u,v})}
        sender_message = singular_direction_message(sender_data, receiver_data, edges,
                                                    senders, num_nodes, out_dim, 'sender_message')
        
        # \sum_{v:(v,u)\in E}{f_e(h_u, h_v, x_{u,v})}
        receiver_message = singular_direction_message(sender_data, receiver_data, edges,
                                                      receivers, num_nodes, out_dim, 'receiver_message')
        
        message_out = sender_message + receiver_message
        
        return message_out

def node_prop(nodes, message, config, scope = 'node_prop'):
    with tf.variable_scope(scope):
        concat = tf.concat([nodes, message], axis = 1)
        node_edim = shapes_list(nodes)[-1]
        
        '''
        the inputs to the transformer encoder are concat [num_nodes, message_dim + node_edim]
        the output are [num_nodes, message_dim + node_edim] --> mapping down to [num_nodes, node_edim]
        '''
        attention_out = base_transformer(concat, config, config.num_layers)
        out = mlp(attention_out, 'to_node_dim', node_edim)
        return out

    
def prop_nodes(nodes, edges, senders, receivers, n_node, n_edge, message_dim, num_iter_steps, scope = 'propogation'):
    with tf.variable_scope(scope):
        
        # use simple named tuple
        transconfig = namedtuple('transconfig', ['num_layers', 'num_heads'])
        transconfig.num_layers = 2
        transconfig.num_heads = 3
        for prop_idx in range(num_iter_steps):
            a_v = message_pass(nodes, edges, senders, receivers, n_node,
                               n_edge, message_dim, scope = 'message_pass_{}'.format(prop_idx))
#                 print('{}. Message: {}'.format(prop_idx, a_v))
            nodes = node_prop(nodes, a_v, transconfig, scope = 'prop_block_{}'.format(prop_idx))
#                 print('{}. Nodes: {}'.format(prop_idx, nodes))

        return nodes
    
def make_graph_embedding(nodes, graph_dim, senders, receivers, n_node, edges = None, scope = 'gated_graph_embedding'):
    """
    get graph embedding tenors.
    
    NOTE: there are a few changes here from the original paper. The paper describes graph embedding
    $h_G = \sum_{v \in V} {g_v^G \cdot h_v^G}$
    
    Now there are following possibilities:
        1. We only aggregate over the unique nodes: Difficulty being that in this case, each graph
           will have the same embedding value.
        2. We aggregate over either senders or receivers: This is terrible in case of a tree, where
           most of the values will be receivers
        3. We aggregate over senders and receivers (what is have implemented here)
        
        4. Ideally to get the graph embedding especially in the case where we can have typed edges,
           we would like to add edge embeddings as well. (future case: edges currently is None)
           
    :param nodes: nodes embedding tensor shape [num_node_types, node_edim]; float
    :param graph_dim: embedding dimension for the graph embedding vector
    :param senders: tensor with sender indices [num_edge,]; int
    :param receivers: tensor with receiver indices [num_edge,]; int
    :param n_node: tensor with number of nodes by graph [num_graphs,]; int
    :param edges: edges feature tensor [num_edges, edge_edim]; float (future case)
    :param scope: tf variable scope name
    
    :return: graph embedding tensor [num_graphs, graph_dim]
    """
    with tf.variable_scope(scope):
        # convert to usable values
        num_nodes = tf.reduce_sum(n_node)
        sender_data = tf.gather(nodes, senders)
        receiver_data = tf.gather(nodes, receivers)
        
        # for sender values
        sender_gated = tf.nn.sigmoid(mlp(sender_data, 's_gated_mlp', graph_dim))
        sender_nodes_mlp = mlp(sender_data, 's_nodes_expanded', graph_dim)
        sender_dot_prod = tf.multiply(sender_gated, sender_nodes_mlp)
        
        # for reciever values
        rec_gated = tf.nn.sigmoid(mlp(receiver_data, 'r_gated_mlp', graph_dim))
        rec_nodes_mlp = mlp(receiver_data, 'r_nodes_expanded', graph_dim)
        rec_dot_prod = tf.multiply(rec_gated, rec_nodes_mlp)
        
        # add those two togeather and reduce_values
        segment_idx = tf_repeat(tf.range(num_nodes), n_node)
        graph_emb = tf.math.unsorted_segment_sum(
            sender_dot_prod + rec_dot_prod,
            segment_idx,
            num_nodes
        ) # should we also add edge information ?
        
        return graph_emb