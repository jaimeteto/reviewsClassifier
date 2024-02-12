
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
#import TfidfTransformer
#import CountVectorizer 

stemmer = PorterStemmer()

#LEMMATIZATION
# Initialize the lemmatizer
def train_and_evaluate(data,X_data,Y_data):
	# vectorizer = TfidfVectorizer()
	# vecorized_data = vectorizer.fit_transform(data[:, 0])
	# fold_index =0
	# for i, (train_index, test_index) in enumerate(fold_indices):
	# 	X_train_data = vecorized_data[train_index]
	# 	y_train = data[train_index,1]
	# 	x_test_data = vecorized_data[test_index]
	# 	y_test_data = data[test_index,1]
	
	kfold = KFold(n_splits=10,shuffle=False)
	
    #convert to pandas
	data_df = pd.DataFrame(data, columns=['Feature', 'Label'])
	
	X_data_series = pd.Series(X_data)
	Y_data_series = pd.Series(Y_data)
	avg_accuracy =0
	#print (len(X_data_series))
	for train_index, test_index in kfold.split(data_df):
		print("Train index: ",train_index,"\n")
		print("Test index: ",test_index,"\n")
		#x_train,x_test,y_train,y_test = data_df[train_index, 'Feature'],data_df.loc[test_index, 'Feature'],data_df.loc[train_index,'Label'],data_df.loc[test_index,'Label']
		x_train = data_df.loc[train_index, 'Feature']
		x_test = data_df.loc[test_index, 'Feature']
		y_train = data_df.loc[train_index, 'Label']
		y_test = data_df.loc[test_index, 'Label']
		predictions = k_nearest_neighbors(x_train,y_train, x_test, 10)
		avg_accuracy+=accuracy_metric(y_test.values.flatten(), predictions)

    # for i, (train_index, test_index) in enumerate(fold_indices):
    #     x_train_data = vectorized_data[train_index]
    #     y_train = [data[j][1] for j in train_index]
    #     x_test_data = vectorized_data[test_index]
    #     y_test_data = [data[j][1] for j in test_index]

        # Assuming k_nearest_neighbors, accuracy_metric, and merge_into_two_columns are defined elsewhere
        # x_train_complete = merge_into_two_columns(x_train_data.toarray(), y_train)
        # predictions = k_nearest_neighbors(x_train_complete, x_test_data.toarray(), 10)

        # avg_accuracy+=accuracy_metric(y_test_data, predictions)
        # print(y_test_data)  
	print(avg_accuracy/10)
		
		# Evaluate your model using test_data
		
		# Aggregate performance metrics
		
		# Optionally, you can print or store the performance metrics for each fold
		
	# Optionally, print or store the aggregated performance metrics


# def merge_into_two_columns(array1, array2):
# 	merged_array = np.column_stack((array1, array2))
# 	return merged_array
def merge_into_two_columns(array1, array2):
    num_rows = len(array2)
    merged_array = [[array1[i], array2[i]] for i in range(num_rows)]
    return merged_array

def split_into_10_folds(data):
	kf = KFold(n_splits=10, shuffle=True, random_state=42)
	fold_indices = []
	for train_index, test_index in kf.split(data):
		fold_indices.append((train_index, test_index))
	return fold_indices

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
	#prediction = max(set(output_values.tolist()), key=output_values.tolist().count)

	return prediction 

# kNN Algorithm
def k_nearest_neighbors(train,train_y, test, num_neighbors):
	predictions = list()
	for row in test:
		output = predict(train,train_y, row, num_neighbors)
		predictions.append(output)
	print
	return(predictions)



# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		#print("acttual: "+actual[i]+ "predicted:"+predicted[i])
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

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

					wl = WordNetLemmatizer()
					
					for w in word_tokens:
						if w not in stop_words and len(w)>1 and not w.isdigit():
							
							
							lemmaword = wl.lemmatize(w)
							stemmed_word = stemmer.stem(lemmaword)
							
							text_with_no_stop_words.append(stemmed_word)

					s = " ".join(text_with_no_stop_words)
					docs.append(s)
					#print(text_with_no_stop_words)
					
					
					
					# Print the first value for each row
					
					
					#print(f"Row {idx + 1} - First Value: {first_value}")
					
		   
			#print(docs)
			return docs,labels

			# transforming each document(review) into a vector usging TFIDF       
			#all_features = v.get_feature_names_out()

			# for word in all_features:
			#     index = v.vocabulary_.get(word)
			#     print(f"{word} {v.idf_[index]}")
			# print(v.vocabulary_)
			# print (len(docs))

	except Exception as e:
		print(f"Error reading the CSV file: {e}")

if __name__ == "__main__":
	# Specify the name of your CSV file
	#csv_file_name = "1695505045_186654_new_train.csv"
	csv_file_name = "prueba1.csv"
	csv_test = "prueba2.csv"
	# Call the function to read the CSV file
	x_train, y_train = read_csv_file(csv_file_name)



	#divide the data into 10 folds(9)=training and 1= test 
	vectorizer = TfidfVectorizer()
	vectorized_data = vectorizer.fit_transform(x_train)
	data = merge_into_two_columns(vectorized_data,y_train)
	train_and_evaluate(data,x_train,y_train)

	