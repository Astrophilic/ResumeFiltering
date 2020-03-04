import os
import pandas as pd
import PyPDF2
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pyresparser import ResumeParser

from sklearn.model_selection import train_test_split

Resume_list = []


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
SRE_list=Get_List_of_resumes('SRE')

def processNewFiles(resume):
  pdfFileObj = open(resume, 'rb')
  pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
  # print(pdfReader.numPages)
  # todo run through all page

  pageObj = pdfReader.getPage(0)
  extracted_text = pageObj.extractText()
  # dictionary of lists

  data = ResumeParser(resume, skills_file='skills.csv').get_extracted_data()

  return data


def processFiles(resume_list, current_role_name):
  for resume in resume_list:
    # pdfFileObj = open(resume, 'rb')
    # pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # print(pdfReader.numPages)
    # # todo run through all page
    # pageObj = pdfReader.getPage(0)
    # extracted_text = pageObj.extractText()
    extracted_text=''
    # dictionary of lists
    data = ResumeParser(resume, skills_file='skills.csv').get_extracted_data()
    Resume_list.append({'Resume': extracted_text, 'Job_role': current_role_name, 'skills': data['skills']})
  # print(Resume_list)


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def getFeatureMatrix(data, vocab):
  featureVector = [0 for i in range(len(vocab))]

  for i in range(len(vocab)):
    for word in data['skills']:
      if word.lower() == vocab[i]:
        featureVector[i] += 1

  print('printing feature vector')

  print(pd.DataFrame(featureVector, columns=vocab))

  print('returning')

  return featureVector


def normalize_document(doc):
  # lower case and remove special characters\whitespaces
  doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)

  doc = re.sub('http\S+\s*', ' ', doc)  # remove URLs
  doc = re.sub('RT|cc', ' ', doc)  # remove RT and cc
  doc = re.sub('#\S+', '', doc)  # remove hashtags
  doc = re.sub('@\S+', '  ', doc)  # remove mentions
  doc = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
               doc)  # remove punctuations
  doc = re.sub(r'[^\x00-\x7f]', r' ', doc)
  doc = re.sub('\s+', ' ', doc)  # remove extra whitespace

  doc = doc.lower()
  doc = doc.strip()
  # tokenize document
  tokens = wpt.tokenize(doc)
  # filter stopwords out of document
  filtered_tokens = [token for token in tokens if token not in stop_words]
  # re-create document from filtered tokens
  doc = ' '.join(filtered_tokens)
  return doc


# normalize_corpus = np.vectorize(normalize_document)

processFiles(BE_list, 'BE')

processFiles(AE_list, 'AE')

processFiles(SRE_list,'SRE')

df = pd.DataFrame(Resume_list)

# defining numerical job indices 0 - resumesAE, 1 -BE, 2-SRE

#
# df.loc[df['Job_role']=='resumesAE','Job_role']=int(0)
# df.loc[df['Job_role']=='BE','Job_role']=int(1)
# df.loc[df['Job_role']=='SRE','Job_role']=int(2)

# print(df['skills'])

df['clean_Resume'] = df['Resume'].apply(lambda doc: normalize_document(doc))

df.to_pickle('resolvedSkills')

skillSet = []

for i in range(len(df)):
  curSkillString = (' ').join(df.iloc[i]['skills'])
  skillSet.append(curSkillString)

# for x in df['skills']:
#   skillString=''
#   for y  in x:
#     skillString+=' '
#     skillString+=y
#
#   corpus.append(skillString)
# corpus = np.array(corpus)


# print(corpus)

# norm_corpus = normalize_corpus(corpus)
# print(norm_corpus)


# instantiate a count vectorizer object
count_vector = CountVectorizer()

# fit to our skillset here
count_vector.fit(skillSet)

# printing vocabulary

print(' vocabulary', str(count_vector.vocabulary_))

print('feature names ', count_vector.get_feature_names())

X = count_vector.transform(skillSet)

X = X.toarray()

# cv = CountVectorizer(min_df=0.0, max_df=1.0)
# cv_matrix = cv.fit_transform(norm_corpus)
# cv_matrix = cv_matrix.toarray()
# print("HELLO",cv_matrix)

# X=cv_matrix

y = df['Job_role']

# print(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

# y_train=y_train.astype('int')


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predict Class
y_pred = classifier.predict(X_test)

# print('predicting for a new file')

# new_data=processNewFiles('./TestingResume/ManojResume.pdf')

# print(new_data['skills'])


# vocab= cv.get_feature_names()
# print("hello", new_data, vocab)


# print(classifier.predict(getFeatureMatrix(new_data,vocab)))

# print('done')

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy so far achieved is')

print(accuracy)
