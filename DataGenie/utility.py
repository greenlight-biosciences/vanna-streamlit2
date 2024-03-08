import re


def is_select_statement(s):
    pattern = r"^\s*SELECT\s"
    return bool(re.match(pattern, s, re.IGNORECASE))

def detect_sql_statement_at_line_start(input_string):
    # Regular expression pattern to detect 'WITH' or 'SELECT' at the start of a line, case-insensitive
    pattern = r'^\s*(WITH|SELECT)\b'
    
    # Search the input string for the pattern with the MULTILINE flag to consider each line
    if re.search(pattern, input_string, re.IGNORECASE | re.MULTILINE):
        return True
    else:
        return False

def optionSelector(option):
    dict={'Instruct Changes (Figure):speaking_head_in_silhouette:':'Tell Data GENIE how the figure should be updated','Edit Code Manually :pencil2:':'Edit the Plot code','Instruct Changes (SQL) :rewind:': 'Instruct Data GENIE on how to modify the SQL query','Instruct Changes (SQL):speaking_head_in_silhouette:':'Instruct Data GENIE on how to modify the SQL query'}
    if dict.get(option) is None:
        return 'Ask me a question about your data...'
    else:
        return dict[option]

def returnMsgFrmtForOAI(message:dict=None, message_log:list=None):
    if message["type"] =='markdown':
        message_log.append({"role": message['role'], "content":message["content"]})
    elif message["type"] =='code':
        message_log.append({"role": message['role'], "content":message["content"]})
    elif message["type"] =='sql':
        message_log.append({"role": message['role'], "content":message["sql"]})
    elif message["type"] =='dataframe':
        message_log.append({"role": message['role'], "content":message["df"].head(int(message["nrows"])).to_string()})
    elif message["type"] =='sql-dataframe':
        message_log.append({"role": message['role'], "content":'Here is a the generated SQL query for your question:'})
        message_log.append({"role": message['role'], "content":message["sql"]})
        message_log.append({"role": message['role'], "content":'Data Preview (first 5 rows):'})
        message_log.append({"role": message['role'], "content":message["df"].head(int(message["nrows"])).to_string()})
    elif message["type"] =='figure-code':
        message_log.append({"role": message['role'], "content":'Here is a figure for the data: <img>'})
        message_log.append({"role": message['role'], "content":'Here is the code for the figure:'})
        message_log.append({"role": message['role'], "content":message["code"]})
    elif message["type"] =='figure':
        message_log.append({"role": message['role'], "content":'<img>'})
    elif  message["type"] =='error':
        message_log.append({"role": message['role'], "content":message["content"]})
    else:
        message_log.append({"role": message['role'], "content":message["content"]})
    return message_log