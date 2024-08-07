import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
# import tensorflow_datasets as tfds
from transformers import BertTokenizer, TFBertModel, BertModel
import csv
import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import pickle

data = pd.read_csv('C:/webimagecrawling/ready1.csv')

##    트레인, 테스트 셋 나누기
train_data, test_data = train_test_split(data[['combined', 'label']], test_size=0.2, random_state=42)

##  토큰화
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
tokenizer.save_pretrained("tokenizer")
tokenizer = BertTokenizer.from_pretrained("tokenizer")
max_seq_len = 300

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels

train_X, train_y = convert_examples_to_features(train_data['combined'], train_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)
test_X, test_y = convert_examples_to_features(test_data['combined'], test_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# directory = 'pickle'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# with open('pickle/train_data.pkl', 'wb') as f:
#     pickle.dump((train_X, train_y), f)

# with open('pickle/test_data.pkl', 'wb') as f:
#     pickle.dump((test_X, test_y), f)

# with open('pickle/train_data.pkl', 'rb') as f:
#     train_X, train_y = pickle.load(f)

# with open('pickle/test_data.pkl', 'rb') as f:
#     test_X, test_y = pickle.load(f)

input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
max_seq_len = 300
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.7)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        x = self.dropout(cls_token)
        prediction = self.classifier(x)

        return prediction
    
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/model_checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1
)
# def scheduler(epoch, lr):
#     if epoch < 3:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = TFBertForSequenceClassification("klue/bert-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
model.fit(train_X, train_y, epochs=5, batch_size=4, validation_split=0.2, verbose=1, callbacks=[checkpoint_callback, early_stopping_callback]) # ,callbacks=[early_stopping, checkpoint]
# model.save_model('model_weights')
# model.load_model('model_weights')
model.load_weights('model/model_checkpoint')
results = model.evaluate(test_X, test_y, batch_size=2)
print("test loss, test acc: ", results)