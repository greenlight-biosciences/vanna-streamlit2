import re


def is_select_statement(s):
    pattern = r"^\s*SELECT\s"
    return bool(re.match(pattern, s, re.IGNORECASE))

def optionSelector(option):
    dict={'Instruct Changes (Figure):speaking_head_in_silhouette:':'Tell Data GENIE how the figure should be updated','Edit Code Manually :pencil2:':'Edit the Plot code','Instruct Changes (SQL) :rewind:': 'Instruct Data GENIE on how to modify the SQL query','Instruct Changes (SQL):speaking_head_in_silhouette:':'Instruct Data GENIE on how to modify the SQL query'}
    if dict.get(option) is None:
        return 'Ask me a question about your data...'
    else:
        return dict[option]