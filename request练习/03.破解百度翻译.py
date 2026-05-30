import requests
import json
import pathlib
if __name__ == "__main__":
    post_url='https://fanyi.baidu.com/sug'
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    word=input("请输入要翻译的单词:")
    data={'kw':word}
    print("开始请求")
    response=requests.post(url=post_url,data=data,headers=headers)
    print("请求结束")
    dic_obj=response.json()
    print(dic_obj)
    filename=pathlib.Path(__file__).parent /f"{word}.json"
    with filename.open("w",encoding="utf-8") as fp:
        json.dump(dic_obj,fp=fp,ensure_ascii=False)
    fp.close()
    print("翻译结果已保存")