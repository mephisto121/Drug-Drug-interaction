import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel
Drug1_model = TFAutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM', from_pt = True);
Drug2_model = TFAutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM', from_pt = True);
input_shape = 128
output_shape = 86


def model(input_shape, output_shape):
    input_ids_drug1 = layers.Input(shape=(input_shape,), name = "input_ids_drug1", dtype = 'int32')
    input_ids_drug2 = layers.Input(shape=(input_shape,), name = "input_ids_drug2", dtype = 'int32')

    input_mask_drug1 = layers.Input(shape=(input_shape,), name = "input_mask_drug1")
    input_mask_drug2 = layers.Input(shape=(input_shape,), name = "input_mask_drug2")

    embed1 = Drug1_model([input_ids_drug1, input_mask_drug1])[1]

    embed2 = Drug2_model([input_ids_drug2, input_mask_drug2])[1]
    d1 = layers.Dense(1024, activation = 'relu')(embed1)
    d1 = layers.Dense(512, activation = 'relu')(d1)
    d2 = layers.Dense(1024, activation = 'relu')(embed2)
    d2 = layers.Dense(512, activation = 'relu')(d2)

    dtot = layers.Concatenate()([d1,d2])
    d3 = layers.Dense(1024, activation = 'relu')(dtot)
    d3 = layers.Dense(512, activation = 'relu')(d3)
    d3 = layers.Dense(256, activation = 'relu')(d3)
    d3 = layers.Dense(output_shape, activation = 'softmax')(d3)

    model = tf.keras.models.Model(inputs = [input_ids_drug1,input_ids_drug2,input_mask_drug1,input_mask_drug2], outputs = d3)
    return model

def ready_model():
    model_train = model(input_shape, output_shape)
    model_train.layers[4].trainable = False
    model_train.layers[5].trainable = False
    return model_train