#import all needed library
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
          #refactoring table
          'localField' : '_id',
          #commit table
          'foreignField' : 'commit_id',
          'as' : 'commit'}},{'$match': {
        #remove any instances where the size of the array is = 0 or has no input
        'commit': {
            #$exists will check if the row exists or not
            '$exists': True,
            '$not': {'$size': 0}

        }}},{


    '$unwind': '$commit'
},
#the needed columns will be selected as a result from the previous query
    #any columns are needed in the future can be added here
    {'$project':{
        'labels.documentation_technicaldept_add':1,
        'labels.documentation_technicaldept_remove':1,
        #commit message
        'message':1,
        #type of refactoring
        'commit.type':1,
        #description of refactoring
        'commit.description':1,
        'commit.detection_tool':1

    }}
])
#create array to save the result from mongoDB and convert it to text format
array = []
for x in result:
    #append all the columns in the array
    #any columns are needed in the future can be added here
    array.append({'id':x['_id'] ,'technicaldept_add':x['labels']['documentation_technicaldept_add'],'technicaldept_remove':x['labels']['documentation_technicaldept_remove'],'message':x['message'],'type':x['commit']['type'],'description':x['commit']['description'],'detection_tool':x['commit']['detection_tool']})

    #save the data into csv file
    pd.DataFrame(array).to_csv('result_mongo.csv')

