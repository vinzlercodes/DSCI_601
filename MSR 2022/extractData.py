import pymongo
import pymongo as pymongo
import pandas as pd

#connect with the MongoDb server
result = pymongo.MongoClient('localhost:27017')['smartshark_2_1']['commit']