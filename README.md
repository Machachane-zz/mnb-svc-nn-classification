# mnb-svc-nn-classificatiom
 Machine Learning models: Multinomial Naive Bayes, Support Vector Machines and Neural Network with Softmax Layer

Important concepts: Glob - to look through all the files; 
save data to csv as pandas dataframe; 
CountVectorizer - count of each typeof the word which is in the dataset;
Save the CountVectorizer output to the pkl file (i.e count_vector.pkl);
TfidfVectorizer - to count the frequency in each document. If it occurs a lot of number of times, then it is important. 
If it occurs a lot of number of times across all documents, then the importance decreases;
Find the TFIDF vector of the dataset and save it in a pklmdocument (i.e. tfidf.pkl);
Split train dataset and test dataset using train_test_split; 
Save the model in a pkl file (i.e. nb_model.pkl);
Model - write a random phrase - load models - apply CountVectorizer, apply TfidfVectorizer - predict the category using the method;
Run the prediction - save it in a csv file (i.e. res_bayes.csv) - print the output
According to: https://www.youtube.com/watch?v=HeKchZ1dauM&t=630s