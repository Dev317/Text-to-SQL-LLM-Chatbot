import pandas_gbq
import re

def get_sql(response):
    sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
    return sql_match.group(1) if sql_match else None

def get_dataframe(sql):
    return pandas_gbq.read_gbq(sql)