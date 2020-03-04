import os
import pickle
from matplotlib.gridspec import GridSpec
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pyresparser import ResumeParser
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC, NuSVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


Resume_list = []
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
skillSet = []

def visualize_data(resume):

  targetCounts = resume['Job_role'].value_counts()
  targetLabels = resume['Job_role'].unique()
  # Make square figures and axes
  plt.figure(1, figsize=(25, 25))
  the_grid = GridSpec(2, 2)
  cmap = plt.get_cmap('coolwarm')
  colors = [cmap(i) for i in np.linspace(0, 1, 3)]
  plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
  source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
  plt.show()

def Get_List_of_resumes(category=''):
  resumes = []
  for root, directories, filenames in os.walk('resumes' + category):
    for filename in filenames:
      if not (filename.startswith(".")):
        file = os.path.join(root, filename)
        resumes.append(file)
  return resumes


BE_list = Get_List_of_resumes('BE')
AE_list = Get_List_of_resumes('AE')
SRE_list = Get_List_of_resumes('SRE')


def processFiles(resume_list, current_role_name):
  for resume in resume_list:
    extracted_text = ''
    try:
      data = ResumeParser(resume, skills_file='skills.csv').get_extracted_data()

    except:
      print('issue creating file', resume)
    finally:
      Resume_list.append({'Resume': extracted_text, 'Job_role': current_role_name, 'skills': data['skills']})


# processFiles(BE_list, 'BE')
# processFiles(AE_list, 'AE')
# processFiles(SRE_list,'SRE')
# df = pd.DataFrame(Resume_list)

# Already Pickled
# df.to_pickle('resolvedSkills')

df = pd.read_pickle('resolvedSkills')

# Plot the job_roles
visualize_data(df)

for i in range(len(df)):
  curSkillString = (' ').join(df.iloc[i]['skills'])
  skillSet.append(curSkillString)

with open('skillSetToFitCountVector.pkl', 'wb') as f:
  pickle.dump(skillSet, f)

# instantiate a count vectorizer object
count_vector = CountVectorizer()

# fit to our skillset here
count_vector.fit(skillSet)

#Transform the skillset here
X = count_vector.transform(skillSet)

# Convert the CSV_matrix to Array
X = X.toarray()

# print(' vocabulary', str(count_vector.vocabulary_))
# print('feature names ', count_vector.get_feature_names())

#Training and test data set input
y = df['Job_role']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1)

# Naive Bayes

classifier = OneVsRestClassifier(GaussianNB())
classifier.fit(X_train, y_train)
# Predict Class
y_pred = classifier.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy so far achieved in Naive Bayes', accuracy)

# OneVsRestClassifier


clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))


gnb = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=1)
MNB = MultinomialNB()
BNB = BernoulliNB()
LR = LogisticRegression()
SDG = SGDClassifier()
SVC = SVC()
LSVC = LinearSVC()
NSVC = NuSVC()

# Train our classifier and test predict
gnb.fit(X_train, y_train)
y_test_GNB_model = gnb.predict(X_test)
print("GaussianNB Accuracy :", accuracy_score(y_test, y_test_GNB_model))

KNN.fit(X_train, y_train)
y_test_KNN_model = KNN.predict(X_test)
print("KNN Accuracy :", accuracy_score(y_test, y_test_KNN_model))

MNB.fit(X_train, y_train)
y_test_MNB_model = MNB.predict(X_test)
print("MNB Accuracy :", accuracy_score(y_test, y_test_MNB_model))

BNB.fit(X_train, y_train)
y_test_BNB_model = BNB.predict(X_test)
print("BNB Accuracy :", accuracy_score(y_test, y_test_BNB_model))

LR.fit(X_train, y_train)
y_test_LR_model = LR.predict(X_test)
print("LR Accuracy :", accuracy_score(y_test, y_test_LR_model))

SDG.fit(X_train, y_train)
y_test_SDG_model = SDG.predict(X_test)
print("SDG Accuracy :", accuracy_score(y_test, y_test_SDG_model))

SVC.fit(X_train, y_train)
y_test_SVC_model = SVC.predict(X_test)
print("SVC Accuracy :", accuracy_score(y_test, y_test_SVC_model))

LSVC.fit(X_train, y_train)
y_test_LSVC_model = LSVC.predict(X_test)
print("LSVC Accuracy :", accuracy_score(y_test, y_test_LSVC_model))

NSVC.fit(X_train, y_train)
y_test_NSVC_model = NSVC.predict(X_test)
print("NSVC Accuracy :", accuracy_score(y_test, y_test_NSVC_model))

filename = 'SVC.sav'
pickle.dump(SVC, open(filename, 'wb'))

filename = 'NSVC.sav'
pickle.dump(NSVC, open(filename, 'wb'))
