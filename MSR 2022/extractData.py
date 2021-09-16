__authors__ = 'Abdullah +vinayak '
import pymongo as pymongo
import pandas as pd
import os
import argparse
"""
This code will extract the needed data from MSR dataset that we have uploaded into a mongoDB,
 for the instances that have  technicaldept_add' value is  True we looking for the refactoring 
 type for each instances from refactoring table .
"""
def connect():
    """
    connect the file with mongo db server and MSR dataset that called smartshark_2_1
    """
    #connect with the MongoDb server
    return pymongo.MongoClient('localhost:27017')['smartshark_2_1']
def extract(connect):
    """
    Thi function is responsible to handle the NOsql queries and send it ro MongoDB server
    :param connect:connection established in the connect function
    """
    #look for the instances where technicaldept_add' is  True
    result = connect['commit'].aggregate([
        {'$match': {
            'labels.documentation_technicaldept_add': True}
        }
        #lookup for the refactoring type where commit_id in commit table =  _id i the refactoring table
        ,
        {'$lookup':
         #refactoring table
             {'from': 'refactoring',
              'localField': '_id',
              #commit table
              'foreignField': 'commit_id',
              'as': 'commit'}},
        #remove any instances where the size of the array is = 0 or has no input
        {'$match': {
            'commit': {
                #$exists will check if the row exists or not
                '$exists': True,
                '$not': {'$size': 0}

            }}

        }, {
        }
        , {
            '$unwind': '$commit'

        },

        {'$lookup':
         #vcs_system table
             {'from': 'vcs_system',
              'localField': 'vcs_system_id',
              #commit table
              'foreignField': '_id',
              'as': 'project_url'}},
        #remove any instances where the size of the array is = 0 or has no input
        {'$match': {
            'project_url': {
                #$exists will check if the row exists or not
                '$exists': True,
                '$not': {'$size': 0}

            }}

        }



        , {
            '$unwind': '$project_url'

        },

        {'$project': {
            'project_url.url':1,
            'labels.documentation_technicaldept_add': 1,
            'labels.documentation_technicaldept_remove': 1,
            #commit message
            'message': 1,
            #type of refactoring
            'commit.type': 1,
            'commit.description': 1,
            #description of refactoring
            'commit.detection_tool': 1
        }}
    ])
    return result
def main():
    """
    :return:
    """
    # Create the parser
    my_parser = argparse.ArgumentParser(description='add a file path to the program')
    my_parser.add_argument('--path',
                           type=str,
                           help='the path to export the file')
    # Execute the parse_args() method
    args = my_parser.parse_args()
    if args.path:
        input_path = args.path
    else:
        input_path = None
    #name of the file
    name_of_file = 'result_mongo.csv'
    #check the path if not correct  then take the file name and save it
    try:
        if input_path != None and input_path[-4:] == '.csv':
            name_of_file = input_path
            print(f"Exporting the file into {input_path}")
        else:
            print(f"No arguments is  provided or the arguments is corrupted exporting the result to the default path 'result_mongo.csv'")
    except:
        print(f"the arguments {input_path} is corrupted ")
    name_of_file = 'result_mongo.csv'
    connecter = connect()
    result = extract(connecter)
    #create array to save the result from mongoDB and convert it to text format
    array = []
    for x in result:
        #append all the columns in the array
        #any columns are needed in the future can be added here
        array.append({'url': x['project_url']['url'],'id': x['_id'], 'technicaldept_add': x['labels']['documentation_technicaldept_add'],
                                    'technicaldept_remove': x['labels']['documentation_technicaldept_remove'],
                                    'message': x['message'], 'type': x['commit']['type'], 'description': x['commit']['description'],
                                    'detection_tool': x['commit']['detection_tool']})
        #save the data into csv file
    try:
        pd.DataFrame(array).to_csv(name_of_file)
        print("the file is exported successfully")
    except:
        print("the exportation process is aborted due to an error")


    if __name__ == '__main__':
        main()
