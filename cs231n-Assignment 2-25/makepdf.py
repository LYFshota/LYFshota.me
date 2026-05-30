import argparse
import os
import subprocess

try:
    from PyPDF2 import PdfMerger

    MERGE = True
except ImportError:
    print("找不到 PyPDF2 (Could not find PyPDF2.)。将不对 pdf 文件进行合并 (Leaving pdf files unmerged.)。")
    MERGE = False


def main(files, pdf_name):
    os_args = [
        "jupyter",
        "nbconvert",
        "--log-level",
        "CRITICAL",
        "--to",
        "pdf",
    ]
    for f in files:
        os_args.append(f)
        subprocess.run(os_args)
        os_args.pop()
        print("已创建 PDF (Created PDF) {}。".format(f))
    if MERGE:
        pdfs = [f.split(".")[0] + ".pdf" for f in files]
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(pdf_name)
        merger.close()
        for pdf in pdfs:
            os.remove(pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 我们传入一个显式的 notebook 参数，以便我们可以提供一个排好序的列表
    # 并且产生一个按顺序排列的 PDF 结果。
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename)
