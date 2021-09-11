import pymongo
import pymongo as pymongo
import pandas as pd
"""
This code will extract the needed data from MSR dataset that we have uploaded into a mongoDB,
 for the instances that have  technicaldept_add' value is  True we looking for the refactoring 
 type for each instances from refactoring table .
"""


#connect with the MongoDb server
result = pymongo.MongoClient('localhost:27017')['smartshark_2_1']['commit'].aggregate([
   #look for the instances where technicaldept_add' is  True
    {'$match': {
        'labels.documentation_technicaldept_add': True}
    }
    #lookup for the refactoring type where commit_id in commit table =  _id i the refactoring table
    ,
    {'$lookup':
         {'from' : 'refactoring',
          'localField' : '_id',
          'foreignField' : 'commit_id',
          'as' : 'commit'}},{'$match': {
        'commit': {
            '$exists': True,
            '$not': {'$size': 0}

        }}}])