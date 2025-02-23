import http.client

conn = http.client.HTTPSConnection("chatgpt-42.p.rapidapi.com")

payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"web_access\":false}"

headers = {
    'x-rapidapi-key': "1e56fad3d4msh7bab8226712579fp191893jsn61d384b0f6a2",
    'x-rapidapi-host': "chatgpt-42.p.rapidapi.com",
    'Content-Type': "application/json"
}

conn.request("POST", "/chatgpt", payload, headers)

res = conn.getresponse()
data = res.read()

decoded_data = data.decode("utf-8")
print(decoded_data)

def createResponse(response):
    print(response.split())

createResponse(decoded_data)