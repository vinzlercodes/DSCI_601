__authors__ = 'Abdullah + Vinayak'

import pymongo as pymongo

import pandas as pd

result = pymongo.MongoClient('localhost:27017')['smartshark_2_1']['commit'].aggregate([
   {'$match': {
        'labels.documentation_technicaldept_add': True}
   }
       ,
       {'$lookup':
            {'from' : 'refactoring',
             'localField' : '_id',
             'foreignField' : 'commit_id',
             'as' : 'commit'}},
        {'$match': {
            'commit': {
                '$exists': True,
                '$not': {'$size': 0}

            }}

   },{
        '$unwind': '$commit'
    },

    {'$project':{
        'labels.documentation_technicaldept_add':1,
        'labels.documentation_technicaldept_remove':1,
        'message':1,
        'commit.type':1,
        'commit.description':1,
        'commit.detection_tool':1

    }}


])

array = []
for x in result:
    array.append({'id':x['_id'] ,'technicaldept_add':x['labels']['documentation_technicaldept_add'],'technicaldept_remove':x['labels']['documentation_technicaldept_remove'],'message':x['message'],'type':x['commit']['type'],'description':x['commit']['description'],'detection_tool':x['commit']['detection_tool']})

pd.DataFrame(array).to_csv('result_mongo.csv')
