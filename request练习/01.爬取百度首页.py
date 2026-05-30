import requests
import pathlib
if __name__ =="__main__":
    url="https://baidu.com"
    print("开始请求")
    response=requests.get(url=url)
    print("请求结束")
    page_text=response.text
    print(page_text)
    out_path=pathlib.Path(__file__).parent /"baidu.html"
    with out_path.open("w",encoding ="utf-8") as fp:
        fp.write(page_text)
    print("写入文件结束")