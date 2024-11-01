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

- python main.py <topics_2 path> <Answers path> -o result_bi_2.tsv result_bi_ft_2.tsv result_ce_2.tsv result_ce_ft_2.tsv -f <finetuned bi-encoder path> <finetuned cross-encoder path>

For more advanced usage:

usage: python main.py <topics_path> <Answers_path> <-o <bi_base> <bi_ft> <ce_base> <ce_ft>> [-f <ft_biencoder_path> <ft_ce_path>] [-q <qrel_path>]

arguments:
  <topics_path>           Path to topics.json file
  <Answers_path>          Path to Answers.json file

options: 
  -o <output_filename1> <output_filename2> <output_filename3> <output_filename4>
  -f <ft_biencoder_path> <ft_ce_path>           paths to pre-finetuned models
  -q <qrel_path>            path to qrel file (fine tunes model if -q but not -f)

examples:
  
  will manually fine tune and then provide all listed output files:
  python main.py data/topics.json data/Answers.json -o bi_base.tsv bi_ft.tsv ce_base.tsv ce_ft.tsv -q data/qrel.tsv
