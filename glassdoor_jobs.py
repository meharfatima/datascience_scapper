# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:25:25 2021

@author: ken
url: https://github.com/PlayingNumbers/ds_salary_proj
"""

import pandas as pd
import data_science_scrapper as scp
keyword = "data scientist"
num_jobs = 100
verbose = False
path = "C:/myDoc/MachineLearning/datascience_scapper/chromedriver"
df = scp.get_jobs(keyword, num_jobs, verbose, path)
df.to_csv('glassdoor_jobs.csv', index=False)