# -*- coding: utf-8 -*-

"""
Script for lexical relation classification for masked training
to reproduce the results in paper:
     "No clues, good clues: Out of context Lexical Relation Classification" 
	 
Script usage example:

$python scripts/masked_lrc_train_evaluate.py \
	--train_templates "' <W1> ' <MASK> ' <W2> '"   \
	--model  "roberta-base" \
	--nepochs 10 \
	--dir_output_results "results/" \
	--batch_size 32 \
	--warm_up 0.1 \
	--nrepetitions 1 \
	--dataset "EVALution" \
	--date `date "+%D-%H:%M:%S"` \
	--train_file "datasets/EVALution/train.tsv" \
	--test_file "datasets/EVALution/test.tsv" \
	--val_file "datasets/EVALution/val.tsv" # comment this line, if there is not val dataset
	
Templates used in the paper:
   template TM1:
          "' <W1> ' <MASK> ' <W2> '"
   template TM2:
          " <W1> <MASK> <W2> "
   template TM3:
         "Today, I finally discovered the relation between <W1> and <W2>: <W1> is the <MASK> of <W2>."  

The tests were performed in Google Colab using packages:
  transformers=4.25.1
  datasets=2.8.0
  torch=1.13.1+cu116
"""

import numpy as np
import pandas as pd
import re
import os
import math
import torch
from torch import nn
from random import randint
import argparse
from datetime import datetime
import logging

from sklearn.metrics import top_k_accuracy_score,confusion_matrix, classification_report
from scipy.stats import entropy

from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_metric, load_dataset

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser(description='Train and test models to classify relations.')
parser.add_argument("-ftrain", "--train_file", required=True, help="Path to tab sep text test file: two words and a relation name by line")
parser.add_argument("-fval","--val_file", required=False, help="Path to tab sep text  val file: two words and a relation name by line")
parser.add_argument("-ftest", "--test_file", required=True, help="Path to tab sep text test file: two words and a relation name by line")
parser.add_argument("-ttrain", "--train_templates", required=True, nargs='+', help="List of templates to verbalize two words for train: They should contain <W1> and <W2> to substitute words in a line.")
parser.add_argument("-m", "--model", required=True, help="Model name checkpoint")
parser.add_argument("-e", "--nepochs", required=True, type=int, help="Number training epochs")
parser.add_argument("-o", "--dir_output_results", default="./", help="Directory to save the test results")
parser.add_argument("-rep", "--nrepetitions", default=1, type=int, help="Number of times the experiment is run")
parser.add_argument("-b", "--batch_size", required=True, type=int, help="Batch size")
parser.add_argument("-wup", "--warm_up", required=False, type=float, default=0.0, help="Warm up ratio for training")
parser.add_argument("-data", "--dataset", required=True, help="Name of the dataset for fine-tuning")
parser.add_argument("-params", "--parameters_list", required=False, help="")
parser.add_argument("-d", "--date", required=False, help="Experiment date")
parser.add_argument("-raw", "--raw_model", default=False, type=bool, help="If True, it is used a no trained model. Default: False")

#parameters
args = parser.parse_args()
model_name = args.model
train_templates = args.train_templates
test_templates = train_templates
train_file = args.train_file
test_file = args.test_file
val_file = args.val_file #None
total_repetitions = args.nrepetitions
batch_size = args.batch_size
warm_up = args.warm_up
name_dataset = args.dataset
params = args.parameters_list
date = args.date
output = args.dir_output_results
total_epochs = args.nepochs
warmup_r = args.warm_up
is_raw = args.raw_model

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

exc_message = "Train templates and test templates must be lists of equal size.\nTrain template list contains {:d} templates and test template list contains {:d}"
if len(train_templates) != len(test_templates):
	raise Exception(exc_message.format(len(train_templates), len(test_templates)))


# dictionary with the one token verbalization of the relations
# in the datasets for roberta and bert tokenizers				  
dict_of_rel_verb = {
'bless':{'roberta':{'event':' event', 'mero':' part', 'random':' random', 'coord':' coordinated', 'attri':' attribute', 'hyper':' subclass'}, 
		  'bert':{'event':'event', 'mero':'part', 'random':'random', 'coord':'coordinated', 'attri':'attribute', 'hyper':'minor'}},
'evalution':{'roberta':{'hasa':' contains', 'madeof':' material', 'partof':' part', 'synonym':' equivalent', 'antonym':' contrary', 'hasproperty':' attribute', 'isa':' subclass'}, 
			 'bert':{'hasa':'contains', 'madeof':'material', 'partof':'part', 'synonym':'synonym', 'antonym':'contrary', 'hasproperty':'attribute', 'isa':'minor'}},
'cogalexv':{'roberta':{'part_of':' part', 'random':' random', 'syn':' equivalent', 'ant':' contrary', 'hyper':' subclass'},
			'bert':{'part_of':'part', 'random':'random', 'syn':'synonym', 'ant':'contrary', 'hyper':'minor'}},
'k&h+n':{'roberta':{'mero':' part', 'false':' random', 'sibl':' coordinated', 'hypo':' subclass'}, 
		 'bert':{'mero':'part', 'false':'random', 'sibl':'coordinated', 'hypo':'minor'}},
'root09':{'roberta':{'random':' random', 'coord':' coordinated', 'hyper':' subclass'}, 
		 'bert':{'random':'random', 'coord':'coordinated', 'hyper':'minor'}},
}

# Convert verbalization of relation labels into tokens						   
tokenizer = AutoTokenizer.from_pretrained(model_name)						   
for d in dict_of_rel_verb:
	for m in dict_of_rel_verb[d]:
		for l in dict_of_rel_verb[d][m]:
			dict_of_rel_verb[d][m][l] = tokenizer.tokenize(dict_of_rel_verb[d][m][l])[0]

# find the type of tokenizer
m = "bert" if tokenizer.tokenize("a")[0] == tokenizer.tokenize(" a")[0]  else "roberta"
d = name_dataset.lower()

if not d in dict_of_rel_verb:
	logging.error("Parameter --dataset is not one of: " + str(list(dict_of_rel_verb.keys())))  
else:
	verb_dict = dict_of_rel_verb[d][m]
	print(verb_dict)
	relation_token_ids = np.array(tokenizer.convert_tokens_to_ids([v for k, v in verb_dict.items()]))
	rev_verb_dict = {v:k for k, v in verb_dict.items()}
	
if name_dataset.lower() not in train_file.lower().split("/"):
	print("It seems that --dataset (" + name_dataset + ") does not correspond with --train_file (" + train_file + ").")
	print("It is needed for fine-tuning that relation labels correspond with exactly one token in the tokenization of the verbalization.")
	print("This is controlled in the script with the variable dict_of_rel_verb.")
	print("If --dataset is the one that correspond with --train_file, proceed")	
	print("Are you sure to continue[y/n]?")
	follow = input()
	if follow != 'y':
		print("Finished")
		raise SystemExit
						 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#create output dir, if it does not exist
try:
	os.makedirs(output)
except:
   pass 

def verb_row(row, template, tokenizer):
	"""
    Create a verbalization of a row (a pair of words and
    a relation label) following a template that can contains 
    <W1>, <W2>, <MASK> to substitute source, 
    target words and the one token relation label
    of a tokenizer. If verb_dict is not None, verb_dict is a 
    dictionary that must contains pairs (key, value)
    where key is a relation label, and value is the one token verbalization
    of the relation label uses to sustitute <MASK> in the template.
    
    Args:
      row -- a series with 'source', 'target' and 'rel'
        template -- a string with (possible) <W1>, <W2>, <MASK>
      tokenizer -- a tokenizer with its special tokens
      verb_dict -- dictionary with the verbalizations (values) of 
        the relation labels (keys)
    
    Returns:
      a dictionary, {'verb':verbalization}, with the key 'verb'
      and the verbalization of the row following the template.
    """
	w1=str(row['source'])
	w2 = str(row['target'])
	sentence = re.sub("<W1>", w1, template)
	sentence = re.sub("<W2>", w2, sentence)
	sentence = re.sub("<MASK>", tokenizer.mask_token, sentence)
	return {'sentence':sentence}

def preprocess_function(rows, tokenizer):
	# use tokenizer to get model inputs
	inputs = tokenizer(rows['sentence'], truncation=True, padding=True, max_length=64)
	
	# find the true labels for the masked language objective
	mask_token_index = [l.index(tokenizer.mask_token_id) for l in inputs['input_ids']]
	labels_train = [[-100]*len(l) for l in inputs['input_ids']]
	
	for l,i,w in zip(labels_train,mask_token_index, rows['token_rel']):
		l[i]=tokenizer.convert_tokens_to_ids(w)
	
	# add the true labels to the dictionary	
	inputs['labels'] = labels_train
	return inputs
	
def topk_by_partition(x, topk):
	unorder_topk_idxs = np.argpartition(x, -topk)[:,-topk:]
	unorder_topk_vals = np.take_along_axis(x, unorder_topk_idxs, axis= -1)
	aux_idxs = np.argsort(-unorder_topk_vals)
	ind = np.take_along_axis(unorder_topk_idxs, aux_idxs, axis=-1)
	val = np.take_along_axis(unorder_topk_vals, aux_idxs, axis=-1)
	return ind, val

def masked2classification(predictions, label_ids, relation_token_ids, topk=5):
	s, t = np.where(label_ids != -100)
	# prediction logits for the masked tokens:
	# logits for all tokens in the tokenizer vocabulary
	logits_toks = predictions[s,t,:]
	# get topk token ids for masked tokens
	topk_idxs, _ =  topk_by_partition(logits_toks, topk)
	# get topk tokens
	topk_tokens = [tokenizer.convert_ids_to_tokens(r) for r in topk_idxs]
	# get only prediction logits for the relation verbalization tokens
	logits_toks_labels = logits_toks[:,relation_token_ids]
	real_token_ids = label_ids[s, t]
	pred_token_ids = relation_token_ids[np.argmax(logits_toks_labels, axis = 1)]

	sfmax = nn.Softmax(dim=-1)
	probs = sfmax(torch.tensor(logits_toks_labels))
	probs = probs.numpy()
	return (pred_token_ids, real_token_ids, probs, topk_tokens)

def results_row(row, tokenizer):
	pred = (row['pred_label'])
	gold = (row['real_label'])
	if pred == gold:
	  row['results'] = True
	else:
	  row['results'] = False
	
	toks_s = tokenizer.tokenize(" " + row['source'])
	toks_t = tokenizer.tokenize(" " + row['target'])
	row['toks_source'] = str(toks_s)
	row['toks_target'] = str(toks_t)
	row['n_toks_source'] = len(toks_s)
	row['n_toks_target'] = len(toks_t)
	return (row)

msgFinetuning = '''Starting fine-tuning with: 
  - model: {:s}
  - train file: {:s} 
  - test file: {:s}
  - val file: {:s}
  - train templates: {:s}
  - test templates: {:s}
*****************************************'''
logging.info(msgFinetuning.format(model_name, train_file, test_file, 
		   val_file if val_file != None else "None", 
		   str(train_templates), str(test_templates)))

# PREPARE DATA
# load train/test files to datasets dict. Also load val file, if it exists
data_files = {'train':train_file,'test':test_file}
if val_file != None:
	data_files['val'] = val_file
all_data = load_dataset('csv', data_files=data_files, sep='\t', header=None, names=['source', 'target', 'rel'], keep_default_na=False)
# add column 'token_rel' with the token that verbalizes the relation 'rel'
#all_data = all_data.map(change_label, fn_kwargs={'verb_dict':verb_dict})
all_data = all_data.map(lambda row : {'token_rel':verb_dict[row['rel'].lower()]})

#copy 'rel' column to 'rel_label' column and encode 'rel_'label' to discret class labels (0,1,...)
all_data = all_data.map(lambda row : {'rel_label':row['rel'].lower()})
all_data = all_data.class_encode_column('rel_label')
print(all_data)

#Calculate number of synsets of the words in test dataset
print("Calculating number synsets for words in test dataset....")
source_words = np.unique(np.array(all_data['test']['source']))
target_words = np.unique(np.array(all_data['test']['target']))
all_words = np.unique(np.concatenate([source_words, target_words]))
synsets_dict = {}
for word in all_words:
	synsets_dict[word] = len(wn.synsets(word))

metric_name = 'f1'
metric = load_metric(metric_name)

# seeds to avoid equal trainings
seeds = [randint(1,100) for n in range(total_repetitions)]
while len(set(seeds)) != total_repetitions:
	seeds = [randint(1,100) for n in range(total_repetitions)]
	
print(seeds)

# For masked training, train templates and test templates are equal
# This code is just in case. For the future
for train_template, test_template in zip(train_templates, test_templates):
	for i in range(total_repetitions):
		print("****** Repetition: " + str(i+1) + "/" + str(total_repetitions) + ". Train template: " + train_template)
		model = AutoModelForMaskedLM.from_pretrained(model_name)
		config = model.config
		if is_raw:
			print('Using LM raw model...')
			model = AutoModelForMaskedLM.from_config(config=config)
		
		# verbalize the datasets with template
		all_data = all_data.map(verb_row, fn_kwargs={'tokenizer':tokenizer, 'template':train_template})
		print("**************************************")
		print(all_data['train']['sentence'][0:10])		
		print(all_data['test']['sentence'][0:10])
		print("**************************************")
		# encode data for language model
		encoded_all_data = all_data.map(preprocess_function, batched=True, batch_size=None, fn_kwargs={'tokenizer':tokenizer})
		# separate the splits in datasets dict
		encoded_verb_train = encoded_all_data['train']
		if val_file != None:
			encoded_verb_val = encoded_all_data['val']
		encoded_verb_test = encoded_all_data['test']
		
		encoded_verb_train.set_format("torch")
		if val_file != None:
			encoded_verb_val.set_format("torch")
		encoded_verb_test.set_format("torch") 
		
		print("****************************")
		print(encoded_verb_test)
		print("****************************")

		args_train = TrainingArguments(
			output_dir='my_checkpoints',
			overwrite_output_dir=True,
			evaluation_strategy= "epoch" if val_file != None else "no",
			save_strategy="epoch" if val_file != None else "no",
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size*2,
			optim="adamw_torch",
			learning_rate=2e-5,
			weight_decay=0.01,
			warmup_ratio=warmup_r,
			#fp16=True,
			logging_steps=10,
			load_best_model_at_end=True if val_file != None else False,
			num_train_epochs=total_epochs,
			report_to='all',
			seed=seeds[i],
			save_total_limit = 1
		)	
			
		trainer = Trainer(
			model, #model to train
			args_train,  #arguments to train
			train_dataset=encoded_verb_train,
			eval_dataset = encoded_verb_val if val_file != None else None,
			tokenizer=tokenizer, #it is needed the tokenizer that encoded the data for batch
			)
		
		#start training
		trainer.train() 
		
		# predictions
		pred_token_ids_list = []
		real_token_ids_list = []
		probs_list = []
		topk_tokens_list = []
		def batch(size, i):
			""" Get the i'th batch of the given size """
			return slice(size*i, size*i + size)
		
		print("Start predictions...")
		batch_pred_size = 128
		NUM_LABELS = encoded_verb_test.features['rel_label'].num_classes
		TOPK = 2*NUM_LABELS
		encoded_verb_test.set_format('numpy')
		for j in range(math.ceil(len(encoded_verb_test)/batch_pred_size)):
			data_slice = Dataset.from_dict(encoded_verb_test[batch(batch_pred_size,j)])
			p = trainer.predict(data_slice)
			slice_pred_token_ids, slice_real_token_ids, slice_probs, slice_topk_tokens = masked2classification(p.predictions, p.label_ids, relation_token_ids, topk=TOPK)
			pred_token_ids_list.append(slice_pred_token_ids)
			real_token_ids_list.append(slice_real_token_ids)
			probs_list.append(slice_probs)
			topk_tokens_list.append(slice_topk_tokens)
			
		
		pred_token_ids = np.concatenate(pred_token_ids_list)
		real_token_ids = np.concatenate(real_token_ids_list)
		probs = np.concatenate(probs_list)
		topk_tokens = np.concatenate(topk_tokens_list)
		
		pred_rel_test = [rev_verb_dict[tokenizer.convert_ids_to_tokens([id])[0]] for id in pred_token_ids ]
		real_rel_test = [rev_verb_dict[tokenizer.convert_ids_to_tokens([id])[0]] for id in real_token_ids]
		pred_rel_token_test = [tokenizer.convert_ids_to_tokens([id])[0] for id in pred_token_ids ]
		real_rel_token_test = [tokenizer.convert_ids_to_tokens([id])[0] for id in real_token_ids]
		pred_rel_test_label = encoded_verb_test.features['rel_label'].str2int(pred_rel_test)
		real_rel_test_label = encoded_verb_test['rel_label']
		print(metric.compute(predictions=pred_rel_test_label, references=real_rel_test_label, average='macro'))

		results_acc = (classification_report(real_rel_test, pred_rel_test, digits=4, output_dict=True))
		print(results_acc)
		encoded_verb_test.set_format('numpy')
		results_words = pd.DataFrame({'pred_label':pred_rel_test_label, 'pred_rel':pred_rel_test, 'pred_verb_token': pred_rel_token_test,
									  'real_label':real_rel_test_label, 'real_rel':real_rel_test, 'real_verb_token': real_rel_token_test,
									  'source':encoded_verb_test['source'], 'target':encoded_verb_test['target']})
		results_words = results_words.apply(results_row, axis=1, tokenizer=tokenizer)
		
		#reorder columns in probs to order in labels
		pos_colums_probs_aux = encoded_verb_test.features['rel_label'].str2int([rev_verb_dict[tokenizer.convert_ids_to_tokens([id])[0]] for id in relation_token_ids])
		pos_colums_probs = sorted(range(len(pos_colums_probs_aux)), key=lambda k: pos_colums_probs_aux[k])
		probs_df = pd.DataFrame(probs)
		probs_df = probs_df.iloc[:,pos_colums_probs]
		probs_df.columns = encoded_verb_test.features['rel_label'].names
		# calculate entropy of probs
		chaos = entropy(probs, axis = 1, base = 2)
		chaos_df =  pd.DataFrame(chaos, columns=['entropy'])
		# for number of synsets
		nsynsets = results_words.apply(lambda x : [synsets_dict[x['source']],synsets_dict[x['target']]], axis=1, result_type='expand')
		nsynsets.columns = ['nsynsests_source', 'nsynsests_target']
		# for topk predicted tokens
		columns=["best_tok#" + str(i+1) for i in range(TOPK)]
		topk_tokens_df = pd.DataFrame(topk_tokens, columns=columns)
		
		results_words = pd.concat([results_words, probs_df, chaos_df, nsynsets,topk_tokens_df], axis = 1)
		
		now = datetime.now()
		now = now.strftime('%y-%m-%d_%H-%M-%S')
 
		fname = output + name_dataset + '_I' + str(i).zfill(2) + "_" + now
		print(fname)
		with open((fname + '.txt') , 'w') as f:
			print(vars(args), file=f)
			print(date, file=f)
			print(results_acc, file=f)
			
			results_words.to_csv(fname + '.csv', index=False)		
			
   