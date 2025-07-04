
import os
from pymongo import MongoClient
from dotenv import load_dotenv

def main():
    """Main function"""
    # Load environment variables
    load_dotenv()

    # Get MongoDB URL from environment
    mongodb_url = os.getenv('MONGODB_URL')
    if not mongodb_url:
        print("Error: MONGODB_URL not found in environment variables")
        print("Please create a .env file with MONGODB_URL=your_mongodb_connection_string")
        return

    try:
        client = MongoClient(mongodb_url)
        db = client['waterandfish']
        lessons_collection = db['Lessons']

        import json

        # Find documents where model_data_url is not null and not an empty string
        lessons = list(lessons_collection.find(
            {
                'model_data_url': {'$nin': [None, '']}
            },
            {'sign_text': 1, 'model_data_url': 1, '_id': 0}
        ))

        with open('lessons_with_model_data.json', 'w', encoding='utf-8') as f:
            json.dump(lessons, f, ensure_ascii=False, indent=4)

        print(f"Successfully exported {len(lessons)} lessons to lessons_with_model_data.json")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()
