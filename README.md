# IR-Assignment3

AidanB-
bi-encoder,
dataprocessor

EvanK-
cross-encoder,
evaluation graphs/tables

# Directions

- cd to the directory and run: pip install -r requirements.txt (recommend using venv or conda env since some specific version requirements)
* Use python version 3.12.0


To run specifically for the assignment topics_2 results using pre-finetuned models, run:

python main.py \<topics\_2 path\> \<Answers path\> -o result\_bi\_2.tsv result\_bi\_ft\_2.tsv result\_ce\_2.tsv result\_ce\_ft\_2.tsv -f \<finetuned bi-encoder path\> \<finetuned cross-encoder path\>

For more advanced usage:

usage: python main.py \<topics\_path\> \<Answers\_path\> \<-o \<bi\_base\> \<bi\_ft\> \<ce\_base\> \<ce\_ft\>\> \[-f \<ft\_biencoder\_path\> \<ft\_ce\_path\>\] \[-q \<qrel\_path\>\]

arguments:
  \<topics\_path\>           Path to topics.json file  
  \<Answers\_path\>          Path to Answers.json file

options: 
  -o \<output\_filename1\> \<output\_filename2\> \<output\_filename3\> \<output\_filename4\>  
  -f \<ft\_biencoder\_path\> \<ft\_ce\_path\>           paths to pre-finetuned models  
  -q \<qrel\_path\>            path to qrel file (fine tunes model if -q but not -f)

examples:
  
  will manually fine tune and then provide all listed output files:  
  python main.py data/topics.json data/Answers.json -o bi_base.tsv bi_ft.tsv ce_base.tsv ce_ft.tsv -q data/qrel.tsv
