import pymongo
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
             'as' : 'commit'}},{'$match': {
            'commit': {
                '$exists': True,
                '$not': {'$size': 0}

            }}

   },{
        '$unwind': '$commit'
    },


