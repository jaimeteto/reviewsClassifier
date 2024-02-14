
"""
Created on Tue Jan 30 00:31:19 2024

@author: jaime
"""

import numpy as np
import os
import csv
import re 
import nltk
import random
import pandas as pd 
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_distances
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
 

stemmer = PorterStemmer()
#predicting 
def predict_test_data(data,X_data,Y_data):
	data_df = pd.DataFrame(data, columns=['Feature', 'Label'])
	print(data_df.columns)
	x_train = data_df['Feature']
	y_train = data_df['Label']
	predictions = k_nearest_neighbors(x_train,y_train, Y_data, 19)
	return predictions

#training 	
def train_and_evaluate(data):

	#divide the data into 10 folds(9)=training and 1= test
	kfold = KFold(n_splits=10,shuffle=False)
	
	#convert to pandas for more efficiency
	data_df = pd.DataFrame(data, columns=['Feature', 'Label'])
	avg_accuracy =0
	
	#cross validation using 10 folds
	for train_index, test_index in kfold.split(data_df):
		print("Train index: ",train_index,"\n")
		print("Test index: ",test_index,"\n")
		x_train = data_df.loc[train_index, 'Feature']
		x_test = data_df.loc[test_index, 'Feature']
		y_train = data_df.loc[train_index, 'Label']
		y_test = data_df.loc[test_index, 'Label']
		predictions = k_nearest_neighbors(x_train,y_train, x_test, 5)
		avg_accuracy+=accuracy_metric(y_test.values.flatten(), predictions)

   
	print(avg_accuracy/10)
		
	


def merge_into_two_columns(array1, array2):
	num_rows = len(array2)
	merged_array = [[array1[i], array2[i]] for i in range(num_rows)]
	return merged_array


def cosine_distance(vec1, vec2):
	
	return cosine_distances(vec1, vec2)

#fuction used to pick k nearest neigbors
def get_neighbors(train,train_y, test_row, num_neighbors):
	distances = list()
	index = 0
	for train_row_x in train:
	
		#data_df_complete = train + train_row_y
		#print(test_row.toarray())
		dist = cosine_distance(test_row.toarray(), train_row_x.toarray())
		distances.append((dist,index))
		index +=1
	#sorting the tuple by distance
	distances.sort(key=lambda tup: tup[0])
	neighbors = list()
	
	for i in range(num_neighbors):
		neighbors.append(distances[i][1])
	return neighbors
# Make a prediction with neighbors
def predict(train,train_y, test_row, num_neighbors):
	  # return the num_neigbors
	neighbors = get_neighbors(train,train_y, test_row, num_neighbors)
	  #get the class value from the last column
	output_values = list()
	for neigbor in neighbors:
		# get the class value for those neighbors
		output_values.append(train_y.iloc[neigbor])
	
	prediction = max(set(output_values), key=output_values.count)
	print("prediction",prediction)
	return prediction 

# kNN Algorithm
def k_nearest_neighbors(train,train_y, test, num_neighbors):
	predictions = list()
	index =0
	for row in test:
		print(index)
		output = predict(train,train_y, row, num_neighbors)
		predictions.append(output)
		index +=1
	return(predictions)



# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		#print("acttual: "+actual[i]+ "predicted:"+predicted[i])
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 

#preprocessing for predicting test data
#returns processed test data
def read_csv_file_test(file_name):
	try:
		# Get the current script directory
		script_dir = os.path.dirname(os.path.abspath(__file__))

		# Construct the full path to the CSV file in the same folder as the script
		csv_file_path = os.path.join(script_dir, file_name)

	   # Open the CSV file and read it line by line
		with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			
			stop_words = set(stopwords.words('english'))

			docs =[]

			# Iterate through each row
			for idx, row in enumerate(csv_reader):
				
				# Check if the row is not empty and has at least two columns (assuming 'column1' and 'column2')
				if row and len(row) >= 1:

					# Access the text column
					text = ",".join(row[0:])  # Join remaining columns into a single text

					#removing punctuation from text column
					text_with_no_ln = text.replace("\\n", "")
					lowercase_string = text_with_no_ln.lower()

					text_with_no_punctutation = re.sub(r'[^\w\s]', ' ', lowercase_string)
					
					#tokenize each word
					word_tokens= word_tokenize(text_with_no_punctutation)
					
					#remove all stop words
					text_with_no_stop_words = []
					
					#LEMMATIZATION
					# Initialize the lemmatizer
					wl = WordNetLemmatizer()
					
					for w in word_tokens:
						if w not in stop_words and len(w)>1 and not w.isdigit():
							
							
							lemmaword = wl.lemmatize(w)
							stemmed_word = stemmer.stem(lemmaword)
							
							text_with_no_stop_words.append(stemmed_word)

					s = " ".join(text_with_no_stop_words)
					docs.append(s)
			return docs
					
		   
			#print(docs)
	except Exception as e:
		print(f"Error reading the CSV file: {e}")		
	
    
#preprocessing for training data
#returns prepocessed data and their respective class
def read_csv_file(file_name):  
	try:
		# Get the current script directory
		script_dir = os.path.dirname(os.path.abspath(__file__))

		# Construct the full path to the CSV file in the same folder as the script
		csv_file_path = os.path.join(script_dir, file_name)

	   # Open the CSV file and read it line by line
		with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			
			stop_words = set(stopwords.words('english'))

			docs =[]
			
			labels =[]

			# Iterate through each row
			for idx, row in enumerate(csv_reader):
				
				# Check if the row is not empty and has at least two columns (assuming 'column1' and 'column2')
				if row and len(row) >= 2:
					# Access the first column (integer with rating)
					first_value = row[0]

					labels.append(first_value)

					# Access the text column
					text = ",".join(row[1:])  # Join remaining columns into a single text

					#removing punctuation from text column
					text_with_no_ln = text.replace("\\n", "")
					lowercase_string = text_with_no_ln.lower()

					text_with_no_punctutation = re.sub(r'[^\w\s]', ' ', lowercase_string)
					
					#tokenize each word
					word_tokens= word_tokenize(text_with_no_punctutation)
					
					#remove all stop words
					text_with_no_stop_words = []
					
					#LEMMATIZATION
					# Initialize the lemmatizer
					wl = WordNetLemmatizer()
					
					for w in word_tokens:
						if w not in stop_words and len(w)>1 and not w.isdigit():
							
							
							lemmaword = wl.lemmatize(w)
							stemmed_word = stemmer.stem(lemmaword)
							
							text_with_no_stop_words.append(stemmed_word)

					s = " ".join(text_with_no_stop_words)
					docs.append(s)
			
			return docs,labels

	except Exception as e:
		print(f"Error reading the CSV file: {e}")

def read_csv_column(filename,col_index):
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(filename,header=None)
    
    # Extract the specified column as a list
    column_data = data_df.iloc[:, col_index].tolist()
    
    return column_data

if __name__ == "__main__":

	# Specify the name of your CSV file
	#csv_file_name = "1695505045_186654_new_train.csv"
	csv_file_name = "prueba5.csv" #prueba5 is a csv document containing 1000 docs
	csv_test = "1695505045_2169955_new_test.csv"
	#csv_test = "prueba3.csv"
	csv_results = "1695505045_2513769_format.csv"

	# Call this function to read the CSV file and preprocess
	x_train, y_train = read_csv_file(csv_file_name)
	
    #process Test Data
	test_data = read_csv_file_test(csv_test)
	
	# transforming each document(review) into a vector usging TFIDF    
	vectorizer = TfidfVectorizer()
	vectorized_data = vectorizer.fit_transform(x_train)
	vocabulary = vectorizer.vocabulary_
	vocabulary_size = len(vocabulary)

	# Print the vocabulary and its size
	print("Vocabulary:")
	print(vocabulary)
	print("\nVocabulary Size:", vocabulary_size)
	test_vectorized_data = vectorizer.transform(test_data)

	#uncomment this when training and testing on trainig data
	#data = merge_into_two_columns(vectorized_data,y_train)
	#train_and_evaluate(data)

	#uncommnet this when predicting
	data = merge_into_two_columns(vectorized_data,y_train)
	predictions =predict_test_data(data,x_train,test_vectorized_data)
	#output file
	csv_file_path = 'output.csv'
	#Writing data to the CSV file
	with open(csv_file_path, mode='w', newline='') as file:
		writer = csv.writer(file)
		for item in predictions:
			writer.writerow([item])
	

	