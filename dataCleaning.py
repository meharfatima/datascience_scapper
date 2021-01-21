# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:55:16 2021

@author: FatimMe
"""

import pandas as pd
df = pd.read_csv('glassdoor_jobs.csv')
df = df.drop(df.columns[[0]], axis=1)
"remove row with no salary"
df = df[df['Salary Estimate']!= '-1']
"Parse Salary data"
df['employer_provided_salary']=df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)
df['per_hour']=df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
min_kd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))
min_hr_emp = min_kd.apply(lambda x: x.replace('Employer Provided Salary:', '').replace(' Per Hour', ''))
df['min_salary']=min_hr_emp.apply(lambda x: int(x.split('-')[0]))
df['max_salary']=min_hr_emp.apply(lambda x: int(x.split('-')[1]))
df['avg_salary']=(df.min_salary+df.max_salary)/2
"parse location to state"
df['company']=df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)
df['state']=df['Location'].apply(lambda x: x.split(',')[1])
df['same_state']=df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis =1)
"age of company"
df['age_of_company']=df.Founded.apply(lambda x: x if x < 1 else 2021-x)
"Parse job description"
df['python']=df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['R']=df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)
df['spark']=df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['sql']=df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['aws']=df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['excel']=df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

"save"
df.to_csv('salary_data_cleaned.csv', index=False)