import urllib.request
from datetime import datetime
import requests

def download_image(url: str) -> str:
    """Download image from URL and save it to path."""
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    name = "logs/query_image_" + date + ".jpg"

    # try:
    #     urllib.request.urlretrieve(url, name)
    # except:
    #     return ""
    with open(name, 'wb') as handle:
        response = requests.get(url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)
    return name

