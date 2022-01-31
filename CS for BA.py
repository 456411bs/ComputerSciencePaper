
"""""
@author Bonnestiksma
"""""
import json
import random
import re
from collections import defaultdict
import nltk
from random import shuffle
import itertools
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from IPython.display import display
import numpy as np
from scipy.spatial.distance import pdist, jaccard
import pandas as pd

nltk.download('punkt')
random = random.seed(0)


with open('TVs-all-merged.json') as f:
   data = json.load(f)
   update = {'TVS' :{}}
   update['TVS'].update(data)
   Model_Words = []
   Model_Words_Total = []
   MW = [None] * 1624 # number of titles on websites
   i = 0
   info_matrix = pd.DataFrame(np.zeros((1624,3)), columns = ['Model','Shop','Brand'])
   brands = ["samsung","philips","lg","toshiba","nec","haier","sharp","viewsonic", "sony", "coby","optoma",
          "tcl","hisense","sansui","sceptre","panasonic","proscan","naxa","vizio","jvc", "sanyo", "insignia","magnavox"
          ,"rca","venturer","westinghouse","supersonic"]
   for TV in update['TVS']:
      for Model in update['TVS'][TV]:
         info_matrix.iloc[i,1] = Model['shop']
         info_matrix.iloc[i,0] = Model['modelID']
         Model['title'] = Model['title'].lower()
         Model['title'] = Model['title'].replace("'","inch").replace('"',"inch").replace(' "',"inch").replace('”',"inch").replace(' ”',"inch").replace("-inch","inch").replace(" inch","inch").replace("inches","inch").replace(" inches","inch").replace("hertz","hz").replace("-hz","hz").replace(" hz","hz").replace("-","").replace("/","").replace(",","").replace(";","").replace(":","").replace("(","").replace(")","").replace("amazon.com","").replace("bestbuy.com","").replace("thenerds.net","").replace("newegg.com","").replace("best buy","")
         y =  re.findall("([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)", Model['title'])
         for brand in brands:
            if brand in Model['title']:
               info_matrix.iloc[i,2] = brand
               break
         Model_Words = [y[i][0] for i in range(len(y))]
         Model_Words_Total += Model_Words
         MW[i] = Model_Words
         i += 1
Model_Words_Total = sorted(set(Model_Words_Total))
def one_hot_matrix(index,MW):
   one_hot = [None] * len(index)
   Model_Words_Total =[]
   MW_new =[None] * len(index)
   j = 0
   for i in index:
      Model_Words_Total += MW[i]
      MW_new[j] = MW[i]
      j +=1
   Model_Words_Total = sorted(set(Model_Words_Total))  # wellicht nog brands en types
   # obtain real duplicates
   real_duplicates = []
   new_info_matrix = info_matrix.loc[index]
   md = np.unique(new_info_matrix['Model'])
   for model in md:
      if list(new_info_matrix['Model']).count(model) > 1:
         real_duplicates.append(list(itertools.combinations(list(new_info_matrix[new_info_matrix['Model'] == model].index), 2)))
   for i in range(len(MW_new)):
      one_hot[i] = list([1 if x in MW_new[i] else 0 for x in Model_Words_Total])
   hot_matrix = np.array(one_hot)
   x,y = hot_matrix.shape
   return(hot_matrix,real_duplicates,Model_Words_Total,x,new_info_matrix)


#building multiple minhash vectors
def building_minhash_function(vocab_n, n):
   hashes = []
   for _ in range(n):
      hash_vector = list(range(1, vocab_n + 1))
      shuffle(hash_vector)
      hashes.append(hash_vector)
   return hashes

# Creating signatures

def Create_Signatures(matrix):
   sign_matrix = np.zeros((x,N),dtype=int)
   for i in range(0,x):
      index = np.nonzero(matrix[i])[0].tolist()
      row_nrs = minhash[:, index]
      sign = np.min(row_nrs, axis=1)
      sign_matrix[i] = sign
   return sign_matrix

def Candidate_Pairs(sign_matrix,bands):
   X, Y = sign_matrix.shape
   hash_buckets = defaultdict(set)
   bands = np.array_split(sign_matrix, bands, axis=0)
   for i, band in enumerate(bands):
      for j in range(Y):
         band_id = tuple(list(band[:, j]) + [str(i)])
         hash_buckets[band_id].add(j)
   candidate_pairs = set()
   for bucket in hash_buckets.values():
      if len(bucket) > 1:
         for pair in itertools.combinations(bucket, 2):
            candidate_pairs.add(pair)
   return candidate_pairs

def dissim_matrix(candidate_pairs):
   dissim_matrix = np.ones((x,x)) * 100000 # set all to 'infinity'

   for candidates in candidate_pairs:
      if ((IF_matrix.iloc[candidates[0], 1] != IF_matrix.iloc[candidates[1], 1]) & (IF_matrix.iloc[candidates[0], 2] == IF_matrix.iloc[candidates[1], 2])):
         dissim_matrix[candidates[0],candidates[1]] = pdist([Signature_Matrix[candidates[0],:],Signature_Matrix[candidates[1],:]], 'jaccard')
         dissim_matrix[candidates[1],candidates[0]] = pdist([Signature_Matrix[candidates[0],:],Signature_Matrix[candidates[1],:]], 'jaccard')
   return dissim_matrix

def duplicates(clusters):
   all_duplicates = [[] for _ in range(clusters.n_clusters_)]
   for i, label in enumerate(cluster.labels_):
      if list(cluster.labels_).count(label) > 1:
         all_duplicates[label].append(i)
   all_duplicates = [ a for a in all_duplicates if a != []]
   final_duplicates = []
   for pair in all_duplicates:
      final_duplicates.append(list(itertools.combinations(pair, 2)))
   return final_duplicates

def calculate_measures(duplicates,candidate_pairs,real_duplicates):
   n_comparisons =0
   total_ids = []
   total_duplicates =0
   identified_duplicates =0
   for predict in duplicates:
      for i in predict:
         if IF_matrix.iloc[list(i)[0]]["Model"] == IF_matrix.iloc[list(i)[1]]["Model"]:
            identified_duplicates +=1
         n_comparisons += 1
   for true in real_duplicates:
      for j in true:
         total_duplicates += 1
   print(total_duplicates)
   comparison_fraction = n_comparisons/len(list(itertools.combinations(range(x),2)))
   PQ = identified_duplicates/n_comparisons
   PC = identified_duplicates/total_duplicates
   f1_measure = 2*((PQ*PC)/(PQ+PC))
   return(f1_measure,PQ,PC,comparison_fraction)




#main part

#define variables
N = round(len(Model_Words_Total) / 2) # number of minhashes 50% of total modelwords excluding full 100%
bands = [[2,308],[8,77],[77,8],[308,2]] #total of 616 number of minhashes
distance_threshold = 0.5
N_bootstraps = 5
percentage_bootstraps = 0.63


result_matrix = np.zeros((N_bootstraps+1,len(bands)),dtype=object)
for bootstrap in range(N_bootstraps):
   # splitting data
   training_data = info_matrix.sample(frac=percentage_bootstraps, random_state=0)
   testing_data = info_matrix.drop(training_data.index)

   # obtaining hot_matrix + other values
   hot_matrix_calculate = one_hot_matrix(list(training_data.index), MW)
   hot_matrix = hot_matrix_calculate[0]
   real_duplicates = hot_matrix_calculate[1]
   Model_Words_Total = hot_matrix_calculate[2]
   x = hot_matrix_calculate[3]
   IF_matrix = hot_matrix_calculate[4]

   i= 0
   for band_size in bands:

      # obtaining signature matrix
      minhash = np.array(building_minhash_function(len(Model_Words_Total), N))  # reduce signature by 50%
      Signature_Matrix = Create_Signatures(hot_matrix)

      # get candidate pairs and make dissimilarity matrix
      c_pairs = list(Candidate_Pairs(Signature_Matrix.transpose(), band_size[0]))
      diss_matrix = dissim_matrix(c_pairs)

      # cluster and get predicted pairs
      cluster = AgglomerativeClustering(distance_threshold=distance_threshold, affinity='precomputed', linkage='single', n_clusters=None)
      cluster.fit(diss_matrix)
      predicted_duplicates = duplicates(cluster)
      #calculate performance measures
      measures_calculate = calculate_measures(predicted_duplicates,c_pairs,real_duplicates)
      f1 = measures_calculate[0]
      PQ = measures_calculate[1]
      PC = measures_calculate[2]
      comparison_fraction = measures_calculate[3]

      result_matrix[bootstrap][i] = [f1,PQ,PC,comparison_fraction]
      i += 1
   print("bootstrap")
results = pd.DataFrame(result_matrix,columns=[str(x) for x in bands])
print(results.iloc[:,0])
print(results.iloc[:,1])
print(results.iloc[:,2])
print(results.iloc[:,3])
print(results.iloc[:,4])
print(results.iloc[:,5])
print(results.iloc[:,6])
print(results.iloc[:,7])
#df = pd.DataFrame({   'name': ['alice','bob','bob','bob','charlie','charlie','charlie'],'age': [25,26,12,23,45,67,89]})
#pff = pd.DataFrame({   'name': [['alice'],['bob','charlie'],[],["d"]],'age': [None,26,27,0],"iets":[None,[],[],None]})
#dff = np.zeros((4,4))
#listd = ['alice','bob']
#total = ['bob','charlie','fred','alice']
#dff[0] = total.map(lambda x: 1 if x in listd else 0)
#pff.insert(0,'yo',None)
#pff["new"] = pff['iets'].map(lambda x: x if x is not None else pff["name"][pff.index[pff["iets"] == x].tolist()[0]])
#df.at[1,'yo'] =["hallo"] +["ddd"]
#df.iloc[1,1] = 2
#words = re.findall(r'() ', ' dames iets-anders ')
#eywas = ["hallo"] +["ddd"]
#df.dtypes
#df.loc[]
#list(df.index[df['name']=='bob'])[0]
#pff["hell"] = pff["name"] + pff["iets"]
#x = np.zeros((170000,170000))
#startimes = [x.value_counts() for x in df['age']]
#startimes = [list(df.index[df['name']== x])[0].last_index() for x in df['name']]
#startimes = [x.value_counts() for x in df['age']]
