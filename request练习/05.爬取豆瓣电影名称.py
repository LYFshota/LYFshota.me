import requests
import pathlib
import json
from bs4 import BeautifulSoup
if __name__ == "__main__":
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    for start_num in range(0,100,20):
        get_url=f'https://movie.douban.com/top250?start={start_num}'
        response=requests.get(url=get_url,headers=headers)
        html_str=response.text
        soup=BeautifulSoup(html_str,'html.parser')
        all_titles=soup.find_all('span',attrs={'class':'title'})
        for title in all_titles:
                title_string=title.string
                if '/'not in title_string:
                    print(title_string)

