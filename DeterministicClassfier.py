from testNewResume import processNewFiles
import pandas as pd


def getFlatList(skillBag):
  flatSet = set()

  for skillList in skillBag:
    for skill in skillList:
      flatSet.add(skill)

  return flatSet


df = pd.read_pickle('resolvedSkills')

# create separate lists for each job profile


BE_skill_bag = df[df['Job_role'] == 'BE']

SRE_skill_bag = df[df['Job_role'] == 'SRE']

AE_skill_bag = df[df['Job_role'] == 'AE']

BE_skill_set = getFlatList(BE_skill_bag['skills'])

SRE_skill_set = getFlatList(SRE_skill_bag['skills'])

AE_skill_set = getFlatList(AE_skill_bag['skills'])

candidate_skillset = processNewFiles(
  '/Users/pankajhazra/PycharmProjects/ResumeClassifierFi/TestingResume/ManojResume.pdf')

candidate_skillset = set(candidate_skillset)


def getLen(variable):
  return len(variable)


BE_common_skills = BE_skill_set.intersection(candidate_skillset)

AE_common_skills = AE_skill_set.intersection(candidate_skillset)

SRE_common_skills = SRE_skill_set.intersection(candidate_skillset)

Job_role = ['BE', 'AE', 'SRE']

len_BE_common_skills = getLen(BE_common_skills)

len_AE_common_skills = getLen(AE_common_skills)

len_SRE_common_skills = getLen(SRE_common_skills)

len_List = [len_BE_common_skills, len_AE_common_skills, len_SRE_common_skills]

max_Len = max(len_List)

predicted_role = -1

for i in range(len(len_List)):
  if len_List[i] == max_Len:
    predicted_role = i
    break

print('Prediction using skill Matching')
print(Job_role[predicted_role])
