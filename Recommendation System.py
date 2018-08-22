
# coding: utf-8

# In[2]:


import gzip
from collections import defaultdict
import numpy as np
from numpy.linalg import inv
import pandas as pd



# In[23]:


def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
allUsers = []
allItems = []


userRatings = defaultdict(list)
count=1
limit = 100
rating_matrix = np.zeros((limit+1, limit+1))


# In[24]:


for l in readGz("train.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    allUsers.append(user)
    allItems.append(item)
    allRatings.append(l['rating'])
    userRatings[user].append(l['rating'])
    rating_matrix[count][count] = 0
    
    if(count >= limit):
        break
    count += 1


# In[25]:


userRatings
allUsers = set(allUsers)
allItems = set(allItems)
allUsers = list(allUsers)
allItems = list(allItems)
print(len(allUsers), len(allItems))
rating_matrix = np.zeros((len(allUsers),len(allItems)))


# In[26]:


for l in readGz("train.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    indexOfUser = allUsers.index(user)
    indexOfItem = allItems.index(item)
    rating_matrix[indexOfUser][indexOfItem] = l['rating']
    
    if(count >= limit):
        break
    count += 1


# In[27]:


rating_matrix = np.array(rating_matrix)
print(rating_matrix)
print(rating_matrix.shape[0])


# In[3]:


DataFrame = pd.read_excel("ratings_train.xlsx", header= None)
train_X = np.array(DataFrame)
print(train_X.shape)

DataFrame = pd.read_excel("ratings_validate.xlsx", header= None)
validation_X = np.array(DataFrame)
print(validation_X.shape)


# In[5]:





# In[4]:


nRows = train_X.shape[0]
nColumns = train_X.shape[1]

for i in range(0, nRows):
    for j in range(0, nColumns):
        if(train_X[i][j] == -1):
            train_X[i][j] = 0

            
nRows = validation_X.shape[0]
nColumns = validation_X.shape[1]            
for i in range(0, nRows):
    for j in range(0, nColumns):
        if(validation_X[i][j] == -1):
            validation_X[i][j] = 0
                                


# In[34]:


X = [
    [5, 4, 3, 0, 0],
    [1, 2, 0, 1, 5],
    [5, 5, 0, 4, 0],
    [5, 0, 0, 0, 0],
    [5, 0, 3, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 2, 0, 3, 0],
    [5, 4, 0, 0, 0],
]

RATING1 = np.array(X[0:4])
RATING2 = np.array(X[4:6])
RATING3 = np.array(X[6:8])

#totalUser = rating_matrix.shape[0];

#RATING1 = np.array(rating_matrix[0:int(totalUser*0.6)])
#RATING2 = np.array(rating_matrix[int(totalUser*0.6):int(totalUser*0.8)])
#RATING3 = np.array(rating_matrix[int(totalUser*0.8):int(totalUser*1.0)])

RATING1 = train_X
RATING2 = validation_X
RATING3 = np.array(X[6:8])

print(RATING1.shape, RATING2.shape)

print("part1 ", RATING1,"part2 " ,RATING2, "part3", RATING3)


# In[45]:


RATING1 = train_X[0:500]
RATING2 = validation_X[0:100]
print(RATING1.shape, RATING2.shape)


# In[61]:


#ALS model trainning 
# Vlidation 

sum_error = 0
min_error = np.Infinity
min_k=10
min_lambda = 0.01

k_set = [10, 20, 40]
#k_set = [2, 3, 4, 5]
lambda_set = [0.01,0.1,1.0,10.0]
iterationCount = 10



for k in k_set:
    for lambdaValue in lambda_set:
        
        RATING = RATING1
        USER = np.random.random((RATING.shape[0], k))
        PRODUCT = np.random.random((k, RATING.shape[1]))
        WEIGHT = (RATING>0)
        
        PRODUCT_T = PRODUCT.T
        USER_T = USER.T
        RATING_T = RATING.T
        WEIGHT_T = WEIGHT.T
        print("Traning Set")
        for itr in range(iterationCount):
            for n, weight_n in enumerate(WEIGHT):
                #print(itr, n)
                part1 = np.dot(PRODUCT, np.dot(np.diag(weight_n), PRODUCT.T))
                # K X M , (M X M , M X K)
                # K X M , M X K
                # K X K

                part2 = lambdaValue * np.eye(int(k))
                # K X K
                #print(k)
                #print(part1, part2)
                p1 = part1 + part2
                # K X K

                p1 = np.array(p1)
                p1 = inv(p1)
                # K x K

                p2 = np.dot(PRODUCT, np.dot(np.diag(weight_n), RATING[n].T))
                # K X M, (M X M, M, 1)
                # K X M, M X 1
                # K X 1


                user_n = np.dot(p1, p2)
                # K X K, K X 1
                # K X 1

                USER[n] = user_n.T
                # 1 X K

            USER_T = USER.T

            for m, weight_m in enumerate(WEIGHT_T):
                #print(itr, m)
                part1 = np.dot(USER_T, np.dot(np.diag(weight_m), USER))
                # K X N, (N X N, N X K)
                # K X N, N X K
                # K X K

                part2 = lambdaValue * np.eye(int(k))
                # K X K

                p1 = part1 + part2
                # K X K

                p1 = np.array(p1)
                p1 = inv(p1)
                # K x K

                p2 = np.dot(USER_T, np.dot(np.diag(weight_m), RATING[ : , m]))
                # K X N, (N X N, N X 1)
                # K X N, N X 1
                # K X 1

                product_m = np.dot(p1, p2)
                # K X K, K X 1
                # K X 1

                PRODUCT[: , m] = product_m
                # K x 1

            PRODUCT_T = PRODUCT.T
            
        PREDICTION = np.dot(USER, PRODUCT)
        #def RMSE(Q, Z, W):
        error = RMSE(RATING, PREDICTION, WEIGHT)
        print(k, lambdaValue, error)
        
        
        print("Validation Set")
        
        newUserPREDICTIONMAT = np.zeros(RATING2.shape)
        newUserWeightMAT = np.zeros(RATING2.shape)
        
        # validation
        for count, rate in enumerate(RATING2):
            #newUserRating = np.array ([5, 0, 3, 0, 0])
            newUserRating = rate
            # 1 X M
            newUserWEIGHT = (newUserRating>0)
            newUserWEIGHT = np.array(newUserWEIGHT)
            # 1 X M
            newUserWeightMAT[count] = newUserWEIGHT
            #print(za)


            part1 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), PRODUCT.T))
            # K X M , (M X M , M X K)
            # K X M , M X K
            # K X K

            part2 = lambdaValue * np.eye(int(k))
            # K X K
            
            p1 = part1 + part2
            # K X K

            p1 = np.array(p1)
            p1 = inv(p1)
            # K x K

            p2 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), newUserRating.T))
            # K X M, (M X M, M X 1)
            # K X M, M X 1
            # K X 1

            #print(p2)

            user_n = np.dot(p1, p2)
            # K X K, K X 1
            # K X 1

            #print(user_n) 
            # 1 X K

            newUserPREDICTION = np.dot(user_n, PRODUCT)
            # 1 X K, K X M
            # 1 X M
            newUserPREDICTIONMAT[count] = newUserPREDICTION
            #print(newUserPREDICTION)
            #print(newUserRating)
            #print(sum_error)
            #sum_error += RMSE(newUserRating, newUserPREDICTION, newUserWEIGHT)
        sum_error = RMSE(RATING2, newUserPREDICTIONMAT, newUserWeightMAT)
        if(sum_error<min_error):
            min_error = sum_error
            min_lambda = lambdaValue
            min_k = k

        
        print(k, lambdaValue, sum_error)
        sum_error = 0
    
    
print("Minimum error found: ", min_lambda,min_k, min_error)
k = min_k
lambdaValue = min_lambda

RATING = RATING1
USER = np.random.random((RATING.shape[0], k))
PRODUCT = np.random.random((k, RATING.shape[1]))
WEIGHT = (RATING>0)

PRODUCT_T = PRODUCT.T
USER_T = USER.T
RATING_T = RATING.T
WEIGHT_T = WEIGHT.T

for itr in range(iterationCount):
    for n, weight_n in enumerate(WEIGHT):
        part1 = np.dot(PRODUCT, np.dot(np.diag(weight_n), PRODUCT.T))
        # K X M , (M X M , M X K)
        # K X M , M X K
        # K X K

        part2 = lambdaValue * np.eye(int(k))
        # K X K
        #print(k)
        #print(part1, part2)
        p1 = part1 + part2
        # K X K

        p1 = np.array(p1)
        p1 = inv(p1)
        # K x K

        p2 = np.dot(PRODUCT, np.dot(np.diag(weight_n), RATING[n].T))
        # K X M, (M X M, M, 1)
        # K X M, M X 1
        # K X 1


        user_n = np.dot(p1, p2)
        # K X K, K X 1
        # K X 1

        USER[n] = user_n.T
        # 1 X K

    USER_T = USER.T

    for m, weight_m in enumerate(WEIGHT_T):
        part1 = np.dot(USER_T, np.dot(np.diag(weight_m), USER))
        # K X N, (N X N, N X K)
        # K X N, N X K
        # K X K

        part2 = lambdaValue * np.eye(int(k))
        # K X K

        p1 = part1 + part2
        # K X K

        p1 = np.array(p1)
        p1 = inv(p1)
        # K x K

        p2 = np.dot(USER_T, np.dot(np.diag(weight_m), RATING[ : , m]))
        # K X N, (N X N, N X 1)
        # K X N, N X 1
        # K X 1

        product_m = np.dot(p1, p2)
        # K X K, K X 1
        # K X 1

        PRODUCT[: , m] = product_m
        # K x 1

    PRODUCT_T = PRODUCT.T
PREDICTION = np.dot(USER, PRODUCT)
#print(PREDICTION)
#print(RATING1)


# In[59]:


k = min_k
lambdaValue = min_lambda

print("Started Trainning with ", k, lambdaValue)

RATING = RATING1
USER = np.random.random((RATING.shape[0], k))
PRODUCT = np.random.random((k, RATING.shape[1]))
WEIGHT = (RATING>0)

PRODUCT_T = PRODUCT.T
USER_T = USER.T
RATING_T = RATING.T
WEIGHT_T = WEIGHT.T

for itr in range(20):
    for n, weight_n in enumerate(WEIGHT):
        #print(itr, n)
        part1 = np.dot(PRODUCT, np.dot(np.diag(weight_n), PRODUCT.T))
        # K X M , (M X M , M X K)
        # K X M , M X K
        # K X K

        part2 = lambdaValue * np.eye(int(k))
        # K X K
        #print(k)
        #print(part1, part2)
        p1 = part1 + part2
        # K X K

        p1 = np.array(p1)
        p1 = inv(p1)
        # K x K

        p2 = np.dot(PRODUCT, np.dot(np.diag(weight_n), RATING[n].T))
        # K X M, (M X M, M, 1)
        # K X M, M X 1
        # K X 1


        user_n = np.dot(p1, p2)
        # K X K, K X 1
        # K X 1

        USER[n] = user_n.T
        # 1 X K

    USER_T = USER.T

    for m, weight_m in enumerate(WEIGHT_T):
        #print(itr, m)
        part1 = np.dot(USER_T, np.dot(np.diag(weight_m), USER))
        # K X N, (N X N, N X K)
        # K X N, N X K
        # K X K

        part2 = lambdaValue * np.eye(int(k))
        # K X K

        p1 = part1 + part2
        # K X K

        p1 = np.array(p1)
        p1 = inv(p1)
        # K x K

        p2 = np.dot(USER_T, np.dot(np.diag(weight_m), RATING[ : , m]))
        # K X N, (N X N, N X 1)
        # K X N, N X 1
        # K X 1

        product_m = np.dot(p1, p2)
        # K X K, K X 1
        # K X 1

        PRODUCT[: , m] = product_m
        # K x 1

    PRODUCT_T = PRODUCT.T
PREDICTION = np.dot(USER, PRODUCT)
sum_error = RMSE(RATING1, PREDICTION, WEIGHT)
print(min_k, min_lambda, sum_error)

#print(PREDICTION)
#print(RATING1)


# In[53]:


# Test 

k = min_k
lambdaValue = min_lambda

#RATING = RATING3
#USER = np.random.random((RATING3.shape[0], k))
#PRODUCT = np.random.random((k, RATING3.shape[1]))
#WEIGHT = (RATING3>0)

#PRODUCT_T = PRODUCT.T
#USER_T = USER.T
#RATING_T = RATING.T
#WEIGHT_T = WEIGHT.T

RATING3 =RATING2
sum_error = 0

newUserPREDICTIONMAT = np.zeros(RATING3.shape)
newUserWeightMAT = np.zeros(RATING3.shape)
for count, rate in enumerate(RATING3):

    #newUserRating = np.array ([5, 0, 3, 0, 0])
    newUserRating = rate
    # 1 X M
    newUserWEIGHT = (newUserRating>0)
    newUserWEIGHT = np.array(newUserWEIGHT)
    # 1 X M
    newUserWeightMAT[count] = newUserWEIGHT

    #print(za)


    part1 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), PRODUCT.T))
    # K X M , (M X M , M X K)
    # K X M , M X K
    # K X K

    part2 = lambdaValue * np.eye(int(k))
    # K X K

    p1 = part1 + part2
    # K X K

    p1 = np.array(p1)
    p1 = inv(p1)
    # K x K

    p2 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), newUserRating.T))
    # K X M, (M X M, M X 1)
    # K X M, M X 1
    # K X 1

    #print(p2)

    user_n = np.dot(p1, p2)
    # K X K, K X 1
    # K X 1

    #print(user_n) 
    # 1 X K

    newUserPREDICTION = np.dot(user_n, PRODUCT)
    newUserPREDICTIONMAT[count] = newUserPREDICTION
    #print(newUserPREDICTION)
    #print(newUserRating)
sum_error = RMSE(RATING3, newUserPREDICTIONMAT, newUserWeightMAT)

print(sum_error)


# In[60]:


# Test 
print("Testing: ")
k_set = [10, 20, 40]
#k_set = [2, 3, 4, 5]
lambda_set = [0.01,0.1,1.0,10.0]
iterationCount = 2

#for k in k_set:
    #for lambdaValue in lambda_set:
        
k = min_k
lambdaValue = min_lambda
RATING3 =RATING2
sum_error = 0

newUserPREDICTIONMAT = np.zeros(RATING3.shape)
newUserWeightMAT = np.zeros(RATING3.shape)
for count, rate in enumerate(RATING3):

    #newUserRating = np.array ([5, 0, 3, 0, 0])
    newUserRating = rate
    # 1 X M
    newUserWEIGHT = (newUserRating>0)
    newUserWEIGHT = np.array(newUserWEIGHT)
    # 1 X M
    newUserWeightMAT[count] = newUserWEIGHT

    #print(za)


    part1 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), PRODUCT.T))
    # K X M , (M X M , M X K)
    # K X M , M X K
    # K X K

    part2 = lambdaValue * np.eye(int(k))
    # K X K

    p1 = part1 + part2
    # K X K

    p1 = np.array(p1)
    p1 = inv(p1)
    # K x K

    p2 = np.dot(PRODUCT, np.dot(np.diag(newUserWEIGHT), newUserRating.T))
    # K X M, (M X M, M X 1)
    # K X M, M X 1
    # K X 1

    #print(p2)

    user_n = np.dot(p1, p2)
    # K X K, K X 1
    # K X 1

    #print(user_n) 
    # 1 X K

    newUserPREDICTION = np.dot(user_n, PRODUCT)
    newUserPREDICTIONMAT[count] = newUserPREDICTION
    #print(newUserPREDICTION)
    #print(newUserRating)
sum_error = RMSE(RATING3, newUserPREDICTIONMAT, newUserWeightMAT)

print(sum_error)


# In[31]:


# N X M * (N x M - N X M)
def RMSE(Q, Z, W):
    if(W.sum() == 0):
        return 1000
    #print((Q - Z)**2 )
    #print((W * (Q - Z))**2 )
    #print(np.sum(  (W * (Q - Z))**2  ))
    #print(W.sum())
    return np.sqrt( np.sum(  (W * (Q - Z))**2  ) / W.sum() )


# In[32]:


Q = np.array( ([5, 5, 0, 4, 0],  [5, 5, 0, 4, 0]))
Z = np.array(( [5, 5, 0, 3, 0],  [5, 4, 0, 3, 0]))
W = np.array(( [1, 1, 0, 1, 0], [1, 1, 0, 1, 0] ))
print(RMSE(Q, Z, W))

