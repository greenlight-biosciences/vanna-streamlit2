import re


def is_select_statement(s):
    pattern = r"^\s*SELECT\s"
    return bool(re.match(pattern, s, re.IGNORECASE))

