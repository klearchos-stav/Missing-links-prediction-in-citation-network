import random
import csv
import numpy as np

import networkx as nx
from networkx.algorithms import shortest_path_length
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import nltk
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("C:/Users/klest/Desktop/Big Data/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]


with open("C:/Users/klest/Desktop/Big Data/training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open(r"C:\Users\klest\Desktop\Big Data\node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]
# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)


to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.30)))
training_set_reduced = [training_set[i] for i in to_keep]




"""````````````````````GRAPH CREATION```````````````````````````````````````"""

G=nx.Graph()
G.add_nodes_from (IDs)

edgesG = [(element[0],element[1]) for element in training_set if element[2]=="1"]
G.add_edges_from (edgesG)

G.number_of_edges()
    

G.number_of_nodes()

list(G.nodes)
list(G.neighbors('205220'))
list(G.neighbors('9803085'))
# randomly select 5% of training set

# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []


# number of common neighbors
comm_neigh = []

# no_edge betwwen source and target
no_edge = []

overlap_abs_train=[]
counter = 0
i=0
J=[]
cs_title=[]

cs_abs=[]
target_abs_string=[]
sources_abs_string=[]

every_two=[]
features_TFIDF=[]
sources_title_string=[]
target_title_string=[]
target_title_string=[]
every_two2=[]
cos_sim_train=[]
cos_sim_title_train=[]

tr_short_path=[]

for i in range(len(training_set_reduced)):
    
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_abs = source_info[5].lower().split(" ")
    source_abs = [token for token in source_abs if token not in stpwds]
    source_abs = [stemmer.stem(token) for token in source_abs]
    
    target_abs = target_info[5].lower().split(" ")
    target_abs = [token for token in target_abs if token not in stpwds]
    target_abs = [stemmer.stem(token) for token in target_abs]
    
    source_neigh = G.neighbors(source)
    target_neigh = G.neighbors(target)
    comm_neigh.append(len(set(source_neigh).intersection(set(target_neigh))))
    
      # cosine similarity of titles text
           
    
              
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    overlap_abs_train.append(len(set(source_abs).intersection(set(target_abs))))

   # no_edge.append(nx.edge_disjoint_paths(G, source, target))
    
    preds = nx.jaccard_coefficient(G, [(source,target)])
    for u, v, p in preds:
        J.append(p)
    
    target_abs_string=" ".join(target_abs)
    sources_abs_string=" ".join(source_abs)
    every_two=[sources_abs_string,target_abs_string]
    features_TFIDF = TfidfVectorizer().fit_transform(every_two)
    cos_sim_train.append(cosine_similarity(features_TFIDF)[0,1])
    
    sources_title_string=" ".join(source_title)
    target_title_string=" ".join(target_title)
    every_two2=[sources_title_string,target_title_string]
    titles_TFIDF = TfidfVectorizer().fit_transform(every_two)
    cos_sim_title_train.append(cosine_similarity(titles_TFIDF)[0,1])
    
    
    try:
        tr_short_path.append(shortest_path_length(G,source,target))
    except nx.NetworkXNoPath:
        tr_short_path.append(-1) 
        
     
    counter+=1
    print(counter)
   
   


   
# convert list of lists into array

training_features=[]
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
#training_features = np.array([temp_diff,comm_auth,comm_neigh,cos_sim_train,cos_sim_title_train]).T
training_features = np.array([overlap_title,temp_diff,comm_auth,overlap_abs_train,comm_neigh]).T



# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

"""CREATING OUR algorithm for testin accuracy"""
X_train,X_test,y_train,y_test = train_test_split(training_features,labels_array,test_size=0.3)




a=[]
# train
clf=[]

for i in range(50,501,50):
    clf = GradientBoostingClassifier(n_estimators=i)
    #clf = GradientBoostingClassifier(n_estimators=150)
    #clf = (n_estimators=150)
    
    clf.fit(X_train, y_train)
    predictions_SGD = np.array(clf.predict(X_test))
    print( i, accuracy_score(y_test, predictions_SGD))
    
# test
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
overlap_abs_test=[]
comm_neigh_test = []  
no_edge_test = []
 
counter = 0
J2=[]
target_abs_string_test=[]
sources_abs_string_test=[]
every_two_test=[]
features_TFIDF_test=[]
cos_sim_train_test=[]
sources_title_string_test=[]
target_title_string_test=[]
every_two2_test=[]
titles_TFIDF_test=[]
cos_sim_title_train_test=[]
tr_short_path_test=[]

for i in range(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
     
    source_abs = source_info[5].lower().split(" ")
    source_abs = [token for token in source_abs if token not in stpwds]
    source_abs = [stemmer.stem(token) for token in source_abs]
    
    target_abs = target_info[5].lower().split(" ")
    target_abs = [token for token in target_abs if token not in stpwds]
    target_abs = [stemmer.stem(token) for token in target_abs]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_neigh = G.neighbors(source)
    target_neigh = G.neighbors(target)
    comm_neigh_test.append(len(set(source_neigh).intersection(set(target_neigh))))
    
   # no_edge_test.append(len(list(nx.edge_disjoint_paths(G,source,target)))) 
    
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    overlap_abs_test.append(len(set(source_abs).intersection(set(target_abs))))

    preds = nx.jaccard_coefficient(G, [(source,target)])
    for u, v, p in preds:
        J2.append(p)
        
    target_abs_string_test=" ".join(target_abs)
    sources_abs_string_test=" ".join(source_abs)
    every_two_test=[sources_abs_string,target_abs_string]
    features_TFIDF_test= TfidfVectorizer().fit_transform(every_two)
    cos_sim_train_test.append(cosine_similarity(features_TFIDF)[0,1])
    
    sources_title_string_test=" ".join(source_title)
    target_title_string_test=" ".join(target_title)
    every_two2_test=[sources_title_string,target_title_string]
    titles_TFIDF_test= TfidfVectorizer().fit_transform(every_two)
    cos_sim_title_train_test.append(cosine_similarity(titles_TFIDF)[0,1])
    
      
    try:
        tr_short_path_test.append(shortest_path_length(G,source,target))
    except nx.NetworkXNoPath:
        tr_short_path_test.append(-1) 
    
    counter += 1
    print (counter)
        
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features=[]
testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test,overlap_abs_test,comm_neigh_test]).T
#training_features = np.array([overlap_title,temp_diff,comm_auth,overlap_abs_train,comm_neigh,J,cos_sim_train,cos_sim_title_train]).T

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
#predictions_SGD = list(clf.predict(testing_features))
clf2=[]
clf2 = GradientBoostingClassifier(n_estimators=300)
clf2.fit(training_features, labels_array)
# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions2 = list(clf2.predict(testing_features))
predictions2 = zip(range(len(testing_set)), predictions2)


with open("C:/Users/klest/Desktop/Big Data/GBC_30_5f_not_shortpath_pred_150.csv","w",newline='') as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["id","category"])
    for row in predictions2:
        csv_out.writerow(row)
        
        