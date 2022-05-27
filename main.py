from ast import Break
import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import RobertaTokenizer
import argparse
from model import ready_model
import warnings
warnings.filterwarnings('ignore')


input_shape = 128
output_shape=68
model_predict = ready_model()

model_predict.load_weights('model/')
labels = pd.read_csv('data\labels.csv')

seq_len = 128
tokenizer = RobertaTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM');


def tokenize(sentence, tokenizer):
    input_ids, input_masks = [],[]

    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=seq_len, pad_to_max_length=True, 
                                            return_attention_mask=True, return_token_type_ids=False)
    input_ids.append(inputs['input_ids'])
    input_masks.append(inputs['attention_mask'])
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

def parse_args():

    parser = argparse.ArgumentParser(description='NLP model for DDI prediction from the command line')

    parser.add_argument('-d1', '--drug1', default=None, type=str, 
                        help='SMILES input for drug.no1')
    parser.add_argument('-d2', '--drug2', default=None, type=str,
                        help='SMILES input for drug.no2')
    parser.add_argument('-n', '--number_of_outputs', default=4, type=int,
                        help="The number of predicted outputs with the highest probability")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if bool(args.drug1) == True:
        if Chem.MolFromSmiles(args.drug1) !=None:
            d1_ids, d1_mask = tokenize(args.drug1, tokenizer=tokenizer)
        else:
            print('Invalid SMILES string for Drug.no1')
            Break
    else:
        print('Empty input for Drug.no1')
        Break

    if bool(args.drug2) == True:
        if Chem.MolFromSmiles(args.drug2) !=None:
            d2_ids, d2_mask = tokenize(args.drug2, tokenizer=tokenizer)
        else:
            print('Invalid SMILES string for Drug.no2')
            Break
    else:
        print('Empty input for Drug.no2')
        Break
    
    predict = model_predict.predict([d1_ids, d2_ids, d1_mask, d2_mask])
    number = tf.argmax(predict, axis = 1)
    print(labels['label_name'][number])
