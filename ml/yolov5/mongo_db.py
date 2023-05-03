import pymongo
from fuzzywuzzy import fuzz
"""
Run the following command from the yolov5 folder

docker run -d \
  --name robocop-mongo \
  -p 27017:27017 \
  -v ./mongo_data:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  -e MONGO_INITDB_DATABASE=license_plates \
  mongo:latest


"""


# set up the MongoDB client
client = pymongo.MongoClient("mongodb://admin:secret@localhost:27017/")


# get the database and collection
db = client["license_plates"]
collection = db["license_plates_entries"]

def initial_entries():
    entries = [
        {"license_plate": "ABC123", "name": "John Doe"},
        {"license_plate": "DEF456", "name": "Jane Smith"},
        {"license_plate": "GHI789", "name": "Bob Johnson"},
        {"license_plate": "C063122", "name": "Charbel Bou Maroun"}
    ]

    # insert the entries into the collection
    collection.insert_many(entries)


def get_name_from_license_plate(license_text):
    documents = list(collection.find())
    license_plates_dict = {}
    for document in documents:
        license_plates_dict[document["license_plate"]] = document["name"]
    # find the closest match using fuzzy string matching
    best_match = None
    best_ratio = -1
    for license_plate in list(license_plates_dict.keys()):
        ratio = fuzz.ratio(license_text, license_plate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = license_plate

    # if a match was found with a ratio greater than 70 (arbitrary threshold), print the name
    if best_ratio > 70:
        print(f"Best match is {best_match}")
        return license_plates_dict[best_match]
    else:
        return None