import os
import tensorflow.Keras as K
import numpy as np

# Function to construct CNN Model
def build_cnn_model(reg_coeff=0, dropout_rate=0):
    """
    Constructs a LSTM+CNN model 
    
    Returns:
      model: Keras model built with functional API
    """

    relu = K.layers.ReLU()
    kernel_reg = K.regularizers.l2(reg_coeff)
    input = K.Input(shape = data_train[0].shape)
    
    #### CONV UNIT 1 ####
    x = K.layers.Conv2D(filters = 320, 
                        kernel_size = (1,26), 
                        strides=(1, 1),
                        padding='same',
                        activation='relu', 
                        use_bias=True, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros', 
                        kernel_regularizer=kernel_reg)(input)

    x = K.layers.MaxPool2D(pool_size=(1,13), 
                            strides=(1,13), 
                            padding='same')(x)

    ##### DROPOUT ######
    x = K.layers.Dropout(dropout_rate)(x)

    ##### BI-DIRECTIONAL LSTM #####
    target_shape = x.shape[-2:]
    x = K.layers.Reshape(target_shape)(x)
    forward_lstm = K.layers.LSTM(320, 
                                 activation="tanh", 
                                 return_sequences=True,
                                 recurrent_activation="sigmoid",
                                 use_bias=True, 
                                 recurrent_regularizer=K.regularizers.l2(reg_coeff))
    backward_lstm = K.layers.LSTM(320, 
                                  go_backwards=True,
                                  activation="tanh", 
                                  return_sequences=True,
                                  recurrent_activation="sigmoid",
                                  use_bias=True, 
                                  recurrent_regularizer=K.regularizers.l2(reg_coeff))
    x = K.layers.Bidirectional(forward_lstm, 
                               backward_layer=backward_lstm,
                               merge_mode='concat')(x)

    ##### DROPOUT #####
    x = K.layers.Dropout(dropout_rate)(x)

    ##### FC & OUTPUT #####
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(925, activation='relu')(x)
    pred = K.layers.Dense(1, activation='sigmoid')(x)

    model = K.Model(inputs=input, outputs=pred)
    return model
    

# Function to Compile and Train Model
def cnn_train(training_dataset,
              validation_dataset,
              reg_coeff,
              dropout_rate,
              learning_rate,
              batch_size, 
              num_epochs,
              num_training=None,
              num_validation=None,
              do_save_model=False, 
              model_dir=None,
              verbose=False): 

    # build model, define loss fxn and optimizer
    model = build_cnn_model(reg_coeff, dropout_rate)
    loss = K.losses.BinaryCrossentropy()
    opt = K.optimizers.RMSprop(learning_rate = learning_rate)

    # if train or validation length not explicitly passed, use all elements
    num_training = (training_dataset[0].shape[0] 
            if num_training is None else num_training)

    num_validation = (validation_dataset[0].shape[0]
            if num_validation is None else num_validation)

    x_train, y_train = training_dataset
    x_val, y_val = validation_dataset

    # number of batches to run during each function call
    num_batches = int(np.floor(x_train.shape[0]/batch_size))

    # define callbacks for early stop
    early_stopping_monitor = K.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

    # define callbacks for model checkpoints
    checkpoint_path=os.path.join(model_dir,'chkpt_models')
    checkpoint = K.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, 
                                            'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                            monitor='val_loss',
                                            verbose = 1)

# Load checkpoint:
    def get_init_epoch(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        #filename = os.path.splitext(filename)[0]
        init_epoch = filename.split("_")[1]
        print("init_epoch", init_epoch)
        return int(init_epoch)

    if checkpoint_path is not None:
        # Load model:
        model = K.models.load_model(checkpoint_path)
        # Finding the epoch index from which we are resuming
        initial_epoch = get_init_epoch(checkpoint_path)
    else:
        model.compile(loss = loss, 
                    optimizer = opt, 
                    metrics = ['accuracy',
                                'AUC'], 
                    steps_per_execution = num_batches)
        initial_epoch = 0

    # train model
    history = model.fit(x = x_train,
            y = y_train,
            batch_size = batch_size,
            epochs = num_epochs,
            validation_data=(x_val, y_val),
            # validation_split = 0.2,
            verbose = verbose,
            callbacks=[early_stopping_monitor,checkpoint],
            initial_epoch=initial_epoch)  

    # save model
    if do_save_model:
        model.save(model_dir) 

    return model, history.history