You should install OpenPrompt https://github.com/thunlp/OpenPrompt
Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

First, Predict initial pseudo-label for the target domain from source domain data.

Second, Adjust target domain data pseudo-label, divide target domain data and iterative training model.

Then, Tote invariant label, Voting invariant label intersection.

Final, Average the experimental results three times.

Note that the file paths should be changed according to the running environment. 

example shell scripts:

python fewshot.py --result_file ./output_fewshot.txt --dataset newsgroups1 --template_id 0 --seed 144 --shot 20 --verbalizer manual

python adjust.py

python div_data.py

python itera_model.py

python voted_label.py

python final.py

