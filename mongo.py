from pymongo import MongoClient


def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["suspect_db"]
    db["suspects"].create_index("name", unique=True)
    db["logs"].create_index("timestamp")
    db["logs"].create_index("name")
    return db


def get_suspect_collection():
    db = get_db()
    return db["suspects"]


def get_log_collection():
    db = get_db()
    return db["logs"]
