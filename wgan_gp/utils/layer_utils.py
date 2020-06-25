from tensorflow.keras import layers

def make_activation_layer(activation):
    if isinstance(activation, dict):
        return activation['factory'](**activation.get('kwargs', dict()))
    return layers.Activation(activation)

def make_mlp(input_tensor, output_size, hidden_layers=1, 
        hidden_units=128, hidden_activation='relu', 
        output_activation=None, use_batch_norm=True,
        use_dropout=False, dropout_rate=0.3):
    
    output_tensor = input_tensor
    for _ in range(hidden_layers):
        output_tensor = layers.Dense(hidden_units)(output_tensor)
        if use_batch_norm:
            output_tensor = layers.BatchNormalization()(output_tensor)
        output_tensor = make_activation_layer(hidden_activation)(output_tensor)
        if use_dropout:
            output_tensor = layers.Dropout(dropout_rate)(output_tensor)
    
    output_tensor = layers.Dense(output_size)(output_tensor)
    output_tensor = make_activation_layer(output_activation)(output_tensor)
    return output_tensor
