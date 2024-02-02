# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 00:31:19 2024

@author: jaime
"""

import numpy as np
import os
import csv
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




def read_csv_file(file_name):
    try:
        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the CSV file in the same folder as the script
        csv_file_path = os.path.join(script_dir, file_name)

       # Open the CSV file and read it line by line
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            
            set(stopwords.words('english'))

            # Iterate through each row
            for idx, row in enumerate(csv_reader):
                # Check if the row is not empty and has at least two columns (assuming 'column1' and 'column2')
                if row and len(row) >= 2:
                    # Access the first column (integer with rating)
                    first_value = row[0]

                    # Access the text column
                    text = ",".join(row[1:])  # Join remaining columns into a single text

                    #removing punctuation from text column
                    text_with_no_punctuation = re.sub(r'[^\ws]', " ", text)
                    
                    #tokenize each word
                    word_tokens= word_tokenize(text_with_no_punctuation)
                    
                    #remove all stop words
                    text_with_no_stop_words = []
                    
                    for w in word_tokens:
                            if w in stopwords:
                                text_with_no_stop_words.append(w)
                    print(text_with_no_stop_words)
                    
                    
                    
                    # Print the first value for each row
                    
                    
                    print(f"Row {idx + 1} - First Value: {first_value}")
                    print(text_with_no_punctuation)
                    
       

    except Exception as e:
        print(f"Error reading the CSV file: {e}")

if __name__ == "__main__":
    # Specify the name of your CSV file
    csv_file_name = "prueba1.csv"

    # Call the function to read the CSV file
    read_csv_file(csv_file_name)