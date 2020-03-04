import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from pyresparser import ResumeParser


def processNewFiles(resume):
  data = ResumeParser(resume, skills_file='skills.csv').get_extracted_data()

  return data['skills']


filename = 'SVC.sav'
loaded_model = pickle.load(open(filename, 'rb'))

candidate_skillset = processNewFiles(
  '/Users/pankajhazra/PycharmProjects/ResumeClassifierFi/TestingResume/ManojResume.pdf')

candidate_skillset = (' ').join(candidate_skillset)

count_vector = CountVectorizer()

with open('skillSetToFitCountVector.pkl', 'rb') as f:
  skillSet = pickle.load(f)

# fit to our skillset here
count_vector.fit(skillSet)

vectorized_candidate = count_vector.transform([candidate_skillset])
vectorized_candidate = vectorized_candidate.toarray()

prediction = loaded_model.predict(vectorized_candidate)

print('Prediction using Classifier')
print(prediction)



