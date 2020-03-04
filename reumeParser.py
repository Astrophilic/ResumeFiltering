from pyresparser import ResumeParser
data = ResumeParser('/Users/pankajhazra/PycharmProjects/ResumeClassifierFi/resumesAE/shravani.vanka_Resume.pdf (1).pdf', skills_file='skills.csv').get_extracted_data()

print(data)
print(type(data))




