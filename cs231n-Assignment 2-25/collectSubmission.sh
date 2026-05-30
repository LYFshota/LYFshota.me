#!/bin/bash
# 注意：不要编辑此文件——可能导致提交不完整
set -euo pipefail

CODE=(
	"cs231n/layers.py"
	"cs231n/classifiers/fc_net.py"
	"cs231n/optim.py"
	"cs231n/solver.py"
	"cs231n/classifiers/cnn.py"
  "cs231n/classifiers/rnn_pytorch.py"
)

# 理想情况下，这些笔记本应该
# 按照问题的顺序排列，以便
# 生成的 pdf 文件
# 也是按照问题的顺序排列
NOTEBOOKS=(
	"BatchNormalization.ipynb"
	"Dropout.ipynb"
	"ConvolutionalNetworks.ipynb"
	"PyTorch.ipynb"
  "RNN_Captioning_pytorch.ipynb"
)
FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )

LOCAL_DIR=`pwd`
ASSIGNMENT_NO=1
ZIP_FILENAME="a2_code_submission.zip"
PDF_FILENAME="a2_inline_submission.pdf"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}未找到所需文件 (Required file) ${FILE}，正在退出 (Exiting).${C_E}"
		exit 0
	fi
done

echo -e "### 正在压缩文件 (Zipping file) ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . \( -name "*.py" -o -name "*.pyx" \)) "cs231n/saved" -x "makepdf.py"

echo -e "### 正在创建 PDF (Creating PDFs) ###"
python makepdf.py --notebooks "${NOTEBOOKS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### 完成！请将 ${ZIP_FILENAME} 和 ${PDF_FILENAME} 提交给 Gradescope。 ###"
