Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt
Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

1、通过源域数据预测目标域初始伪标签
可在命令行输入：
python fewshot.py --result_file ./output_fewshot.txt --dataset newsgroups1 --template_id 0 --seed 144 --shot 20 --verbalizer manual

2、调整目标域数据伪标签：
可在命令行输入：
python adjust.py

3、划分目标域数据
可在命令行输入：
python div_data.py

4、迭代训练模型
可在命令行输入：
python itera_model.py

5、投票不变标签
可在命令行输入：
python voted_label.py

6、投票不变标签取交集
可在命令行输入：
python inter_label.py

7、实验结果三次取平均
可在命令行输入：
python final.py

Note that the file paths should be changed according to the running environment. 

The datasets are downloadable via OpenPrompt.
