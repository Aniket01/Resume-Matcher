import os
import pandas as pd
import extractPDF
from extractPDF import extract_text, TextCleaner, KeytermExtractor
from similarity import Similarity

jobcsv_path = r"Data\training_data.csv"
resumecsv_path = r"Data\Resume Dataset\Resume\Resume.csv"
resume_path = r"Data\Resume Dataset\data\data"

def parse_resumes(filepath:str):
    """
    Reads and extracts the resume from the Kaggle Dataset, 
    cleans and processes into pandas dataframe
    Args:
        filepath(str): path to resumes to be parsed and processed
    Returns:
        df(pandas dataframe): Dataframe object containing resume
        filenames and raw text
    """
    columns = ["filename","raw_str"]
    df = pd.DataFrame(columns=columns)
    for file in os.listdir(filepath):
            d = os.path.join(filepath, file)
            resume_str = extract_text(d)
            tc_obj = TextCleaner(resume_str)
            resume_str = tc_obj.clean_text()
            ke_obj = KeytermExtractor(resume_str)
            ke_list = ke_obj.get_keyterms_based_on_textrank()
            ke_ext=''
            for i in range(len(ke_list)):
              ke_ext = ke_ext + ke_list[i][0] + ' '
            row = [file, ke_ext]
            df.loc[len(df)] = row
    return df

def parse_description():
    """
    Reads and extracts job description data from HuggingFace Dataset
    into pandas dataframe
    Args:
        None
    Returns:
        df(pandas dataframe): Dataframe object containing extracted and
        preprocessed job description & position title.
    """
    columns = ["Job_Description", "Position"]
    df = pd.DataFrame(columns=columns)
    job_df = pd.read_csv(r"Data\training_data.csv")
    for i in range(15):
        tc_obj = TextCleaner(job_df.job_description[i])
        ke_obj = KeytermExtractor(tc_obj.clean_text())
        position = job_df.position_title[i]
        ke_list = ke_obj.get_keyterms_based_on_textrank()
        ke_ext=''
        for i in range(len(ke_list)):
            ke_ext = ke_ext + ke_list[i][0] + ' '
        df.loc[len(df)] = [ke_ext, position]
    return df

def match_position(position:str):
    """
    Matches job description position to the sub-category 
    of resume arranged in the dataset folder
    Arg:
        position(str): 
    Returns:
        rank_dict(dict): dictionary of subfolder containing possible
        matching resumes and ranking scores
    """
    resume_categories = os.listdir(resume_path)
    rank_dict = {}
    for category in resume_categories:
        sim = Similarity(position, category)
        rank_dict.update({category:sim.calculate()})
    return rank_dict

def top_five(df, description):
    """

    """
    l = df.shape[0]
    score_dict={}
    for i in range(l):
        sim_obj = Similarity(df.raw_str[i],description)
        score = sim_obj.calculate()
        score_dict.update({df.filename[i]:score})
    score_dict = dict(sorted(score_dict.items(), key=lambda item:item[1],reverse=True))
    topfive_list = list(score_dict.keys())
    topfive_list = topfive_list[:5]
    return topfive_list

