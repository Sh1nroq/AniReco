import requests
from bs4 import BeautifulSoup
import pandas

url = "https://myanimelist.net/anime.php"
page = requests.get(url)

print(page.status_code)
