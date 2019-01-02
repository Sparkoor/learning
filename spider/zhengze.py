import re

str = '[\'\n                    合租 · 恒大名都 3室1厅 600元                  \']'
pattern = re.compile('\[|\]')
ss = re.match(pattern, str)
print(ss.group(0))

aa = str.replace('[\'', '')
bb = aa.replace('\']', '')
print(bb.strip())
