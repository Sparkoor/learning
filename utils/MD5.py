import hashlib

str_mad5 = hashlib.md5('aaaaaaaaaaaaaaaaaaaaa'.encode('utf-8')).hexdigest()
print(str_mad5)
