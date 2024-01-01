from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import ast
import datasets
from tqdm import tqdm
import time

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class LanguageDataset(Dataset):
	"""
	An extension of the Dataset object to:
		- Make training loop cleaner
		- Make ingestion easier from pandas df's
	"""
	def __init__(self, df, tokenizer):
		self.labels = df.columns
		self.data = df.to_dict(orient='records')
		self.tokenizer = tokenizer
		x = self.fittest_max_length(df)  # Fix here
		self.max_length = x

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		x = self.data[idx][self.labels[0]]
		y = self.data[idx][self.labels[1]]
		text = f"{x} | {y}"
		tokens = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
		return tokens,text

	def fittest_max_length(self, df):  # Fix here
		"""
		Smallest power of two larger than the longest term in the data set.
		Important to set up max length to speed training time.
		"""
		max_length = max(len(max(df[self.labels[0]], key=len)), len(max(df[self.labels[1]], key=len)))
		x = 2
		while x < max_length: x = x * 2
		return x

def training():

	data_sample = load_dataset("QuyenAnhDE/Diseases_Symptoms")

	print(data_sample)
	updated_data = [{'Name': item['Name'], 'Symptoms': item['Symptoms']} for item in data_sample['train']]
	df = pd.DataFrame(updated_data)
	df.head(5)
	df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(', ')))
	print(df)
	DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	print('Device!!!!:',DEVICE)
	tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

	# The transformer
	model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)

	model.eval()
	BATCH_SIZE = 8
	# Dataset Prep
	

	# Cast the Huggingface data set as a LanguageDataset we defined above
	data_sample = LanguageDataset(df, tokenizer)

	print('Training sample length!!!')
	print(data_sample.__len__())

	# print(data_sample.__getitem__(2))

	print(data_sample.fittest_max_length(df))
	# Create train, valid
	train_size = int(0.8 * len(data_sample))
	valid_size = len(data_sample) - train_size
	train_data, valid_data = random_split(data_sample, [train_size, valid_size])

	# Make the iterators
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
	valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

	# Set the number of epochs
	num_epochs = 10

	# Training parameters
	batch_size = BATCH_SIZE
	model_name = 'distilgpt2'
	gpu = 0

	# Set the learning rate and loss function
	## CrossEntropyLoss measures how close answers to the truth.
	## More punishing for high confidence wrong answers
	criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id) ##softmax func and negative log likelyhood
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	tokenizer.pad_token = tokenizer.eos_token

	# Init a results dataframe
	results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu',
	                                'training_loss', 'validation_loss', 'epoch_duration_sec'])




	# The training loop
	for epoch in range(num_epochs):
		start_time = time.time()
		model.train()
		epoch_training_loss = 0
		train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs} Batch Size: {batch_size}, Transformer: {model_name}")
		for batch in train_iterator:
			optimizer.zero_grad()
			print(batch)
			inputs = batch['input_ids'].squeeze(1).to(DEVICE)
			targets = inputs.clone()
			outputs = model(input_ids=inputs, labels=targets)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			train_iterator.set_postfix({'Training Loss': loss.item()})
			epoch_training_loss += loss.item()
		avg_epoch_training_loss = epoch_training_loss / len(train_iterator)

		# Validation
		## This line below tells the model to 'stop learning'
		model.eval()
		epoch_validation_loss = 0
		total_loss = 0
		valid_iterator = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
		with torch.no_grad():
			for batch in valid_iterator:
				inputs = batch['input_ids'].squeeze(1).to(DEVICE)
				targets = inputs.clone()
				outputs = model(input_ids=inputs, labels=targets)
				loss = outputs.loss
				total_loss += loss
				valid_iterator.set_postfix({'Validation Loss': loss.item()})
				epoch_validation_loss += loss.item()

		avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

		end_time = time.time()  # End the timer for the epoch
		epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

		new_row = {'transformer': model_name,
				'batch_size': batch_size,
				'gpu': gpu,
				'epoch': epoch+1,
				'training_loss': avg_epoch_training_loss,
				'validation_loss': avg_epoch_validation_loss,
				'epoch_duration_sec': epoch_duration_sec}  # Add epoch_duration to the dataframe

		results.loc[len(results)] = new_row
		print(f"Epoch: {epoch+1}, Validation Loss: {total_loss/len(valid_loader)}")


	torch.save(model, 'Model/SmallMedLM.pt')

    #print("Training Done@@@@@@@@")
	return 'Training of Model Done'

def Inference(input_str):    
	model_dir = './Model/SmallMedLM.pt'

	# Load the model's state dictionary
	model_state_dict = torch.load(model_dir)

	# Create an instance of your model (assuming your model class is MyModel)
	model = MyModel()

	# Load the state dictionary into the model
	model.load_state_dict(model_state_dict)

	# Set the model to evaluation mode (if needed)
	model.eval()
	input_ids = tokenizer.encode(input_str, return_tensors='pt').to(DEVICE)

	output = model.generate(
	    input_ids,
	    max_length=20,
	    num_return_sequences=1,
	    do_sample=True,
	    top_k=8,
	    top_p=0.95,
	    temperature=0.1,
	    repetition_penalty=1.2
	)

	decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
	print('Result',decoded_output.split('|')[1])
	return {'Input_str':input_str,'Resposne':decoded_output.split('|')[1]}


if __name__ == '__main__':
	print(training())
	Inference('Depression')


