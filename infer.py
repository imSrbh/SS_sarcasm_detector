from clearml import Task
import tensorflow as tf
import pickle

#get the task
task = Task.get_task('271c821288014059b7c6590efb67bbaa')

#load the model
transformer_model_path = task.models.data['output'][0].get_local_copy()
model = tf.keras.models.load_model(transformer_model_path)
print(model.summary())

#load the tokenizer
tokenizer_path  = task.artifacts['local file'].get_local_copy()
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

sentence = ["Coworkers At Bathroom Sink Locked In Tense Standoff Over Who Going To Wash Hands Longer", 
            "Spiking U.S. coronavirus cases could force rationing decisions similar to those made in Italy, China."]

#tokenize input sentences
sequences = tokenizer.texts_to_sequences(sentence)
#pad sequences to max_len
padded =tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

res = model.predict(padded)
print(res)
