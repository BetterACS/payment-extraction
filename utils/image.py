import urllib.request
from datetime import datetime


def download_image(url: str) -> str:
    """Download image from URL and save it to path."""

    date = datetime.now().strftime("%Y%m%d%H%M%S")
    name = "query_image_" + date + ".jpg"

    try:
        urllib.request.urlretrieve(url, name)
    except:
        return ""

    return name
