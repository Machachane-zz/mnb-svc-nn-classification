# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:58:28 2020

@author: Machachane
"""


with open( 'news', 'r') as f:
    text = f.read()
    news = text.split('\n\n')
    count = {'sport': 0, 'world': 0, 'us': 0, 'business': 0, 'health': 0, 'entertainment': 0, 'sci_tech': 0}
    for news_item in news:
        lines = news_item.split('\n')
        print(lines[6])
        file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
        count[lines[6]] = count[lines[6]] + 1
        file_to_write.write(news_item)   #Python will convert \n to os.linesep
        file_to_write.close()
        
#-------------------------------------------------------------------------------------------------------------

import pandas
import glob

category_list = ['sport', 'world', 'us', 'business', 'health', 'entertainment', 'sci_tech']
directory_list = ['data/sport/*.txt', 'data/world/*.txt', 'data/us/*.txt', 'data/business/*.txt', 'data/health/*.txt', 'data/entertainment/*.txt', 'data/sci_tech/*.txt']

text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []

for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1], 'flag' : category_list.index(t[6])})

training_data[0]
 
#-------------------------------------------------------------------------------------------------------------

training_data = pandas.DataFrame(training_data, columns = ['data', 'flag'])
training_data.to_csv('train_data.csv', sep=',', encoding='utf-8')
print(training_data.data.shape)

#-------------------------------------------------------------------------------------------------------------

import pickle
from sklearn.feature_extraction.text import CountVectorizer

#Get vector count
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(training_data.data)

#Save word vector
pickle.dump(count_vect.vocabulary_, open('count_vector.pkl', 'wb'))


#-------------------------------------------------------------------------------------------------------------

from sklearn.feature_extraction import TfidfTransformer

#Transform word vector to TF IDF
tfidf_transformer = TfidfTransformer()

#Save TF-IDF
pickle.dump(tfidf_transformer, open('tfidf.pkl', 'wb'))

#-------------------------------------------------------------------------------------------------------------

#Multinomial Naive Bayes Algorithm 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)
X_train, X_test, y_train, y_test = train_test_split(X_train_ttfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)

#Save Model
pickle.dump(clf, open('nb_model.pkl', 'wb'))


#-------------------------------------------------------------------------------------------------------------

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ['sport', 'world', 'us', 'business', 'health', 'entertainment', 'sci_tech']

docs_new = "Apple stock rises by 10 times"
docs_new = [docs_new]

#Load Model
loaded_vec = CountVectorizer(vocabulary=pickle.load(open('count_vector.pkl', 'rb')))
loaded_tfidf = pickle.load(open('tfidf.pkl', 'rb'))
loaded_model = pickle.load(open('nb_model.pkl', 'rb'))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(category_list[predicted[0]])

#-------------------------------------------------------------------------------------------------------------

predicted = loaded_model.predict(X_test)
result_bayes = pandas.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})
result_bayes.to_csv('res_bayes.csv', sep = ',')

for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])
    
#-------------------------------------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test,predicted)
print(confusion_mat)

#-------------------------------------------------------------------------------------------------------------

from sklearn.neural_network import MLPClassifier

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)

#-------------------------------------------------------------------------------------------------------------

pickle.dump(clf_neural, open('softmax.pkl', 'wb'))

predicted = clf_neural.predict(X_test)
result_softmax = pandas.DataFrame({'true_labels': y_test, 'predicted_labels': predicted})
result_softmax.to_csv('res_softmax.csv', sep = ',')

for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])
    
#------------------------------------------------------------------------------------------------------------- 

from sklearn import svm

clf_svm = svm.LinearSVC()

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf.svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open('svm.pkl', 'wb'))

predicted = clf_svm.predict(X_test)
resul_svm = pandas.DataFrame({'true_labels': y_test, 'predicted_labels': predicted})
result_svm.to_csv('res_svm.csv', sep=',')
for predicted_item, result in zip(predicted, y_test):
    print(category_list[predicted_item], ' - ', category_list[result])
    


#-------------------------------------------------------------------------------------------------------------

