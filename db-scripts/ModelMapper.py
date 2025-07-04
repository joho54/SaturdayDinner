#!/usr/bin/env python3
"""
MongoDB Model Mapper Script
Updates Lessons collection with model_data_url based on sign_text matching
"""

import json
import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

def load_mapping_file(file_path="label_model_mapping.json"):
    """Load the label to model mapping from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['mapper']
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        sys.exit(1)
    except KeyError:
        print(f"Error: 'mapper' key not found in {file_path}")
        sys.exit(1)

def connect_to_mongodb(mongodb_url):
    """Connect to MongoDB and return client and database"""
    try:
        client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB")
        print("list of databases", client.list_database_names())
        # Get database name from URL or use default
        db_name = 'waterandfish'
        print("db_name", db_name)
        db = client[db_name]
        
        return client, db
    except ConnectionFailure:
        print("Error: Failed to connect to MongoDB")
        sys.exit(1)
    except ServerSelectionTimeoutError:
        print("Error: MongoDB server selection timeout")
        sys.exit(1)

def update_lessons_with_model_urls(db, mapping):
    """Update Lessons collection with model_data_url"""
    lessons_collection = db['Lessons']
    print("lessons_collection", lessons_collection)
    # Get all lessons
    lessons = list(lessons_collection.find({}))
    print(f"Found {len(lessons)} lessons in database")
    
    updated_count = 0
    not_found_count = 0
    not_found_signs = []
    
    for lesson in lessons:
        sign_text = lesson.get('sign_text', '')
        
        if sign_text in mapping:
            # Update the lesson with model_data_url
            result = lessons_collection.update_one(
                {'_id': lesson['_id']},
                {'$set': {'model_data_url': mapping[sign_text]}}
            )
            
            if result.modified_count > 0:
                updated_count += 1
                print(f"Updated lesson: {sign_text} -> {mapping[sign_text]}")
            else:
                print(f"No changes made for lesson: {sign_text}")
        else:
            not_found_count += 1
            not_found_signs.append(sign_text)
            print(f"Warning: No mapping found for sign_text: '{sign_text}'")
    
    print(f"\nUpdate Summary:")
    print(f"- Successfully updated: {updated_count} lessons")
    print(f"- No mapping found: {not_found_count} lessons")
    
    if not_found_signs:
        print(f"\nLessons without mapping:")
        for sign in not_found_signs:
            print(f"  - '{sign}'")

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB URL from environment
    mongodb_url = os.getenv('MONGODB_URL')
    if not mongodb_url:
        print("Error: MONGODB_URL not found in environment variables")
        print("Please create a .env file with MONGODB_URL=your_mongodb_connection_string")
        sys.exit(1)
    
    print("Loading label to model mapping...")
    mapping = load_mapping_file()
    print(f"Loaded {len(mapping)} mappings")
    
    print("Connecting to MongoDB...")
    client, db = connect_to_mongodb(mongodb_url)
    
    try:
        print("Updating lessons with model URLs...")
        update_lessons_with_model_urls(db, mapping)
        print("Update completed successfully!")
    except Exception as e:
        print(f"Error during update: {e}")
        sys.exit(1)
    finally:
        client.close()
        print("MongoDB connection closed")

if __name__ == "__main__":
    main()
