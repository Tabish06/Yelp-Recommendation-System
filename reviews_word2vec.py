from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix
from pyspark.mllib.linalg import Matrix, Matrices


import re
import string
import numpy as np


file_location = "gs://cloudcomputingbucketproject/yelp-dataset/"
review_df = spark.read.parquet(file_location+'review.parquet')
business_df = spark.read.parquet(file_location+'business.parquet')
#selecting only businesses from NC
business_nc_df = business_df.filter(business_df['state']== "NC")

business_reviews_nc_df = business_nc_df.join(review_df,business_nc_df.business_id == review_df.business_id).drop(review_df.business_id)
#408060 rows
doc = review_df.rdd.map(lambda line: line['text'].split(" "))
word2vec = Word2Vec()
model = word2vec.fit(doc)


w2v_model_path = 'gs://cloudcomputingbucketproject/main_model/data'
lookup = sqlContext.read.parquet(w2v_model_path).alias("lookup")
#https://stackoverflow.com/questions/34448456/using-word2vecmodel-transform-does-not-work-in-map-function
saved_model = Word2VecModel.load(sc,'gs://cloudcomputingbucketproject/main_model/')

lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

#This code will give stars and the vector list. I'll work on the sentiment analysis tomorrow
def word2vecsentence(row):
  regex = re.compile('[%s]' % re.escape(string.punctuation))
  num_features = 100
  feature_vec = np.zeros((num_features, ), dtype='float32')
  n_words = 0
  cleaned_text =regex.sub('', row['text']).lower()
  
  for term in cleaned_text.split(" "):
    vector = lookup_bd.value.get(term)
    if vector != None :
      feature_vec = np.add(feature_vec,vector)
    n_words = n_words +  1
  if n_words > 0 :
    feature_vec = np.divide(feature_vec,n_words)
  feature_vec = feature_vec.tolist()
  sentiment  = -1
  if row['stars'] > 4:
    sentiment = 1
  elif row['stars'] < 2 :
    sentiment = 0 
  return (row['review_id'],int(row['stars']),sentiment,Vectors.dense(feature_vec))

#https://stackoverflow.com/questions/22129943/how-to-calculate-the-sentence-similarity-using-word2vec-model-of-gensim-with-pyt


review_vector_ratings = business_reviews_nc_df.rdd.map(word2vecsentence).toDF(['review_id','stars','sentiment','word2vec'])
review_vector_ratings.first()

# review_vector_ratings.write.parquet(file_location+'review_vectorized_sentiment.parquet')

reviews_vector_sentiment_df = review_vector_ratings.where( (review_vector_ratings.sentiment == 0) | (review_vector_ratings.sentiment == 1 ) )

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,featuresCol = "word2vec",labelCol = "sentiment")

lrModel = lr.fit(reviews_vector_sentiment_df)
trainingSummary = lrModel.summary


accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))


######## LOGISTIC REGRESSION #########
N1 = 50
N2 = 50
N = N1 + N2
D = 2
K = 2

mu1 = [-1, -1]
cov1 = np.eye(2)

mu2 = [2,3]
cov2 = np.eye(2) * 3

mu3 = [5,3]
cov2 = np.eye(2) * 3

#
#  Train Data
#
C1 = np.random.multivariate_normal(mu1, cov1, N1)
C2 = np.random.multivariate_normal(mu2, cov2, N2)



Xtrain = np.vstack((C1, C2))
Ttrain = np.zeros((N, 2))
Ttrain[:50, 0] = 1
Ttrain[50:, 1] = 1

# class LogReg() :

#   def __init__(self):
#       """
#           weight vector w is initialized as None
#       """
#     self.w = None
  

  
#   def softmax(self,z):
#     if not isinstance(z, np.ndarray):
#         z = np.asarray(z)
#     f = np.exp(z) 
#     return f / (np.sum(f, axis=1, keepdims=True) if len(z.shape) == 2 else np.sum(f))

#   # for linear fx
#   def g(self,X, w):
#     return self.softmax(X @ w) 


#   def train(self, X, T,alpha =0.1):
#     niter = 1000
#     N = X.shape[0]
#     X1 = np.hstack((np.ones((N, 1)), X))

#     D = X.shape[1]
#     K = T_transform.shape[1]
#     self.w = np.zeros((D+1,K))
#     for steps in range(niter) :
#       ys = self.g(X1,self.w)
#       self.w += alpha * X1.T @ (T_transform - ys)

    ## TO DO : write a pyspark code to merge ones array 

## X (200,12) 12
## Y (200,3)  3
## W (12,3) 
# ( 1,12) (12,3) = (1,3)
## W = X @ W = (12,3)
## Predictions = X . W = (200,3)


maxiter = 50
x_train_rdd = sc.parallelize(Xtrain)
t_train_rdd = sc.parallelize(Ttrain)

x_train_rdd = x_train_rdd.map(lambda line: Vectors.dense(np.append(1,line)) )
t_train_rdd = t_train_rdd.zipWithIndex().map(lambda (line,index): IndexedRow(index, line) )

indexed_row = x_train_rdd.zipWithIndex().map(lambda (line, index): IndexedRow(index,line.toArray()))
x_mat = IndexedRowMatrix(indexed_row)
t_mat = IndexedRowMatrix(t_train_rdd).toBlockMatrix()

x_transposed = x_mat.toBlockMatrix().transpose()
def softmax(z):
  f = np.exp(z)
  return f / np.sum(f)

alpha = 0.01
w = np.zeros((Xtrain.shape[1]+1)* Ttrain.shape[1])
weight_matrix = Matrices.dense(Xtrain.shape[1]+1, Ttrain.shape[1],w)

for i in range(0,maxiter):
  x_cross_w =  x_mat.multiply(weight_matrix)
  preds = IndexedRowMatrix(x_cross_w.rows.map(lambda row : IndexedRow(row.index, softmax(row.vector.toArray()) ) )).toBlockMatrix()
  gradient = x_transposed.multiply(preds.subtract( t_mat ))
  gradient = IndexedRowMatrix(gradient.toIndexedRowMatrix().rows.map(lambda row : IndexedRow(row.index,alpha * np.array(row.vector.toArray()) )))
  gradient_np = np.array(gradient.toBlockMatrix().toLocalMatrix().toArray())
  weight_matrix = Matrices.dense(Xtrain.shape[1]+1, Ttrain.shape[1],(np.array(weight_matrix.toArray()) + gradient_np).flatten())
  #Back to array x_cross_w.rows.map(lambda x: (x, )).toDF()