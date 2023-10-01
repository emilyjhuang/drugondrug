#!/usr/bin/env python
# coding: utf-8

# In[22]:


get_ipython().system('pip install pandas')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


# In[23]:


X_file_path = 'DRUG23Q1.txt'
y_file_path = 'REAC23Q1.txt'

X_df = pd.read_csv(X_file_path, sep='$')
y_df = pd.read_csv(y_file_path, sep='$')


# In[18]:


drug_data = {}
reaction_data = {}

# Read and process the DRUGQ1 file.
with open('DRUG23Q1.txt', 'r') as drug_file:
    for line in drug_file:
        parts = line.strip().split('$')
        drug_id = parts[0]
        drug_name = parts[5]
        drug_data[drug_id] = drug_name

# Read and process the REACQ1 file.
with open('REAC23Q1.txt', 'r') as reaction_file:
    for line in reaction_file:
        parts = line.strip().split('$')
        drug_id = parts[0]
        reaction = parts[2]
        reaction_data[drug_id] = reaction

# Correlate data and build X and Y lists.
X = []
Y = []

for drug_id, reaction in reaction_data.items():
    if drug_id in drug_data:
        drug_name = drug_data[drug_id]
        X.append(drug_name)
        Y.append(reaction)


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


batch_size = 1000
num_batches = len(X_train_encoded) // batch_size + 1

# Create a Logistic Regression model
model = LogisticRegression()

# Create a generator to yield batches of data
def batch_generator(X, Y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]

# Train the model using batch processing
for X_batch, Y_batch in batch_generator(X_train_encoded, Y_train, batch_size):
    model.fit(X_batch, Y_batch)


# In[ ]:


Y_pred = model.predict(X_test)

# Evaluate the model's performance.
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

# Print a classification report for more detailed metrics.
report = classification_report(Y_test, Y_pred)
print(report)


# In[2]:


df = pd.read_csv("drugs.csv")


# In[3]:


file_path = 'Products.txt'  # Replace with the actual path to your text file
df = pd.read_csv(file_path, sep='\t')

# Extract features (X) and labels (y)
X = df['DrugName']  
y = df['ActiveIngredient'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


file_path = 'Products.txt' 
df = pd.read_csv(file_path, sep='\t')

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.str.cat(y_train, sep=' '))
X_test_vec = vectorizer.transform(X_test['DrugName'] + ' ' + y_test['ActiveIngredient'])

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_vec)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Example usage of the trained model for prediction
new_drug_name = "DrugX"
new_active_ingredients = "IngredientA, IngredientB"
new_data = pd.DataFrame({'DrugName': [new_drug_name], 'ActiveIngredient': [new_active_ingredients]})
new_data_vec = vectorizer.transform(new_data['DrugName'] + ' ' + new_data['ActiveIngredient'])
prediction = model.predict(new_data_vec)
print(f'Predicted Drug-Drug Combination: {prediction[0]}')


# In[ ]:





# In[12]:


X2_file_path = 'DRUG23Q2.txt'
y2_file_path = 'REAC23Q2.txt'

X2_df = pd.read_csv(X2_file_path, sep='$')
y2_df = pd.read_csv(y2_file_path, sep='$')


# In[13]:


drug_data = {}
reaction_data = {}

# Read and process the DRUGQ1 file.
with open('DRUG23Q1.txt', 'r') as drug_file:
    for line in drug_file:
        parts = line.strip().split('$')
        drug_id = parts[0]
        drug_name = parts[5]
        drug_data[drug_id] = drug_name

# Read and process the REACQ1 file.
with open('REAC23Q1.txt', 'r') as reaction_file:
    for line in reaction_file:
        parts = line.strip().split('$')
        drug_id = parts[0]
        reaction = parts[2]
        reaction_data[drug_id] = reaction

# Correlate data and build X and Y lists.
X2 = []
Y2 = []

for drug_id, reaction in reaction_data.items():
    if drug_id in drug_data:
        drug_name = drug_data[drug_id]
        X2.append(drug_name)
        Y2.append(reaction)


# In[14]:


X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[154]:



df = pd.read_csv("purplebook-search-august-data-download.csv", header =3)

# Extract features (X) and labels (y)
A = df['Proprietary Name'] 
Z = df['Strength']

# Split the data into training and testing sets
A_train, A_test, Z_train, Z_test = train_test_split(A, Z, test_size=0.2, random_state=42)


# In[23]:


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train + ' ' + Y_train)
X_test_vec = vectorizer.transform(X_test + ' ' + Y_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_vec)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Example usage of the trained model for prediction
new_drug_name = "DrugX"
new_active_ingredients = "IngredientA, IngredientB"
new_data = pd.DataFrame({'Drug_Name': [new_drug_name], 'Active_Ingredients': [new_active_ingredients]})
new_data_vec = vectorizer.transform(new_data['Drug_Name'] + ' ' + new_data['Active_Ingredients'])
prediction = model.predict(new_data_vec)
print(f'Predicted Drug-Drug Combination: {prediction[0]}')


# In[ ]:


#df = pd.read_csv("drug_names.tsv", delimiter='\t', header = None)
#X = df.iloc[:, 1]
df = pd.read_csv("drug_effects.tsv", delimiter = '\t', header = None)
X = df.iloc[:, 1]
y = df.iloc[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit a TF-IDF vectorizer on drug names
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')


# In[3]:


file_path = 'products_orange.txt'  # Replace with the actual path to your text file
df = pd.read_csv(file_path, sep='~')

# Extract features (X) and labels (y)
A = df['Trade_Name'] 
Z = df['Strength']

# Split the data into training and testing sets
A_train, A_test, Z_train, Z_test = train_test_split(A, Z, test_size=0.2, random_state=42)

