import requests
import json
# Get data from trafikverket.se
BASE_URL = 'https://api.trafikinfo.trafikverket.se/v2/data.json'

# The base Query is structured as an XML string looking like this:

BASE_QUERY = """
<REQUEST>
  <LOGIN authenticationkey="{authenticationkey}" />
  <QUERY objecttype="{objecttype}" schemaversion="1.5" limit="{limit}">
  <FILTER>
    <EQ name="{name}" value="{value}" />
  </FILTER>
  </QUERY>
</REQUEST>
"""

BASE_QUERY_NO_FILTER = """
<REQUEST>
  <LOGIN authenticationkey="{authenticationkey}" />
  <QUERY objecttype="{objecttype}" schemaversion="1.5" limit="{limit}">
  </QUERY>
</REQUEST>
"""

def postRequest(objecttype, limit, name, value, filter=True):
    """Create a request object with the given parameters"""
    if filter:
        query = BASE_QUERY.format(
            authenticationkey=getAPIKey(),
            objecttype=objecttype,
            limit=limit,
            name=name,
            value=value
        )
    else:
        query = BASE_QUERY_NO_FILTER.format(
            authenticationkey=getAPIKey(),
            objecttype=objecttype,
            limit=limit
        )
    
    # Set headers
    headers = {'Content-Type': 'text/xml'}
    
    # Send request
    return requests.post(BASE_URL, data=query, headers=headers)


def getAPIKey():
    with open('.key') as f:
        return f.read().strip()

def prettyPrintJson(response):
    # Pretty print the JSON
    print(json.dumps(response, indent=4, sort_keys=True, ensure_ascii=False))
    
def main():
    # Get all accidents (deviations)
    response = postRequest('Situation', 100, 'Deviation.IconId', 'roadAccident', filter=True)
    
    # Print the response
    prettyPrintJson(response.json())
    

if __name__ == '__main__':
    main()
    
