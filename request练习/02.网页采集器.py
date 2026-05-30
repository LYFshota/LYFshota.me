import requests
import pathlib
if __name__ == "__main__":
    header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'}
    url='https://www.sogou.com/web'
    kw=input("请输入搜索关键字:")
    param={'query':kw}
    print("开始请求")
    response=requests.get(url=url,params=param,headers=header)
    print("请求结束")
    page_text=response.text
    print(page_text)
    out_path=pathlib.Path(__file__).parent /f"{kw}.html"
    with out_path.open("w",encoding ="utf-8") as fp:
        fp.write(page_text)