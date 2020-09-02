import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
pandas2ri.activate()

def featuresAtLevel(n,taxFeatures):
    if n != 0:
        return [x for x in taxFeatures if x.count(".") == n]
    else: 
        return [x for x in taxFeatures if x.count(".") == n and list(taxFeatures).index(x)<1512]
def doLogReg(X_train,y_train,X_test):
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    return logreg.predict(X_test)

readRDS = robjects.r['readRDS']
df = readRDS('biodata/cirrhosisA.RDS')
#print(df.columns[1512:])
print(df.columns[1511])
print(df["study_condition"])
le = preprocessing.LabelEncoder()
y = le.fit_transform(df["study_condition"])
a = 0
for i in featuresAtLevel(0,df.columns):
    print(i)
    print(df[i][6])
    a += df[i][6]
print("a",a)

X = []
for i in range(6):
    X.append(df[featuresAtLevel(i,df.columns)])
print(X[0].shape)
y_train = [0 for _ in range(6)]
y_test = [0 for _ in range(6)]
X_train = [0 for _ in range(6)]
X_test = [0 for _ in range(6)]
for i in range(6):
    X_train[i],X_test[i],y_train[i],y_test[i]=train_test_split(X[i],y,test_size=0.25,random_state=0)
y_pred = []

for i in range(6):
    print(i,len(X_train[i]),len(y_train[i]),len(X_test[i]),len(y_test[i]))
    y_pred.append(doLogReg(X_train[i],y_train[i],X_test[i]))
for i in range(6):
    print(metrics.confusion_matrix(y_test[i], y_pred[i]))


