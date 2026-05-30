import requests
import pathlib
import json
if __name__ == "__main__":
    get_url='https://movie.douban.com/j/chart/top_list'
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    params={'type':17,'interval_id':'100:90','action':'','start':0,'limit':20}
    print("开始请求")
    response=requests.get(url=get_url,params=params,headers=headers)
    print("请求结束")
    list_obj=response.json()
    print(list_obj)
    filename=pathlib.Path(__file__).parent /"豆瓣电影.json"
    with filename.open("w",encoding="utf-8") as fp:
        json.dump(list_obj,fp=fp,ensure_ascii=False)
    fp.close()