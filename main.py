import pandas as pd
import extractPDF, parser, similarity
from extractPDF import  TextCleaner, KeytermExtractor
from parser import parse_resumes, parse_description, match_position, top_five
from similarity import Similarity
resume_path = r"Data\Resume Dataset\data\data"

job_df =  parse_description()
filepath_list = [["" for j in range(5)] for i in range(15)]
for i in range(15):
    columns = ["filename","raw_str"]
    df1 = pd.DataFrame(columns=columns)
    pos_dict = match_position(job_df.Position[i])
    pos_dict = dict(sorted(pos_dict.items(), key=lambda item:item[1],reverse=True))
    pos_list = list(pos_dict.keys())
    pos_list = pos_list[:3]
    for j in range(3):
        filepath_list[i][j] = f"{resume_path}\{pos_list[j]}"
        df2 = parse_resumes(filepath_list[i][j])
        df1 =pd.concat([df1,df2], ignore_index=True)
    result = top_five(df1, job_df.Job_Description[i])
    del df1
    print(result)

