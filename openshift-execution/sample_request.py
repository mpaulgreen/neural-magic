import requests

def test_request():
    url = "http://localhost:5543/v2/models/sentiment_analysis/infer" # Server's port default to 5543
    obj = {"sequences": "Snorlax loves my Tesla!"}

    response = requests.post(url, json=obj)
    print(response.text)
    # {"labels":["positive"],"scores":[0.9965094327926636]}


if __name__ == "__main__":
    test_request()
    