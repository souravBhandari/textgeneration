import spacy
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM , Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from pickle import dump , load
from keras.models import load_model

def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text


nlp=spacy.load('en_core_web_lg',disable=['parser','tagger','ner'])
nlp.max_length= 1198623

def sep_punc(doc_text):
	return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
d=read_file('e:/mobydick.txt')
token=sep_punc(d)
print(len(token))

#passing some words and predicting next word .
train_len = 26 # taking 25 words to predict 26th word..
text_seq = []
for i in range(train_len,len(token)):
	seq=token[i-train_len:i]
	text_seq.append(seq)
print(' '.join(text_seq[0])) # joining words to form sentences..

tokenizer=Tokenizer()
tokenizer.fit_on_texts(text_seq)
seq=tokenizer.texts_to_sequences(text_seq)
for i in seq[0]:
	print(f"{i}:{tokenizer.index_word[i]}")

	#	creating vocab

vocabulary_size=len(tokenizer.word_counts)
print(vocabulary_size)
seq=np.array(seq)
print(seq)
x=seq[:,:-1] # All Rows except last Column.so as i can split it in features and labels
y=seq[:,-1]# last column with all rows 
y = to_categorical(y, num_classes=vocabulary_size+1)
seq_len=x.shape[1]
print(seq_len)
print(x.shape)#seq,words in sent

def create_model(vocabulary_size,seq_len):	
	model=Sequential()
	model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len))
	model.add(LSTM(50,return_sequences=True))
	model.add(LSTM(50))
	model.add(Dense(50,activation='relu'))
	model.add(Dense(vocabulary_size,activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
    
	return model

#model=create_model(vocabulary_size+1,seq_len) 
#model.fit(x,y,batch_size=128,epochs=550,verbose=1)
#model.save('BIG.h5')
#dump(tokenizer,open('BIG','wb'))	
model = load_model('e:/BIG.h5')
tokenizer=load(open('e:/BIG','rb'))
#	 function for generating text

def gen_text(model,tokenizer,seq_len,seed_text,num_gen_words):
    output_text = []
    input_text = seed_text # initial seeding text s
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0] # it retuns a tuple of item	
        pad_encoded = pad_sequences([encoded_text],maxlen=seq_len,truncating='pre') # as if user add a long or short text then it corrects it.
        pred_word_ind = model.predict_classes(pad_encoded,verbose=0)[0]   #pred prob for each words.
        
        pred_word=tokenizer.index_word[pred_word_ind]
        		
        input_text += ' '+pred_word
        output_text.append(pred_word)
    	
    return ' '.join(output_text)

random.seed(101)
random_pick=random.randint(0,len(text_seq))
random_seed_text=text_seq[random_pick]# choosing randomly words
seed_text=' '.join(random_seed_text)
print(seed_text)

gen_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)



