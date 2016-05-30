import json

"""
Read json file and return the corresponding json object for further process

@input file_name : string
@return data : json object
"""
def read_data(file_name):
	data = []
	f = open(file_name, 'r')
	total = 0
	size = 0
	for line in f:
		# filter only 'Type' : 'Implicit'
		obj = json.loads(line)
		if obj['Type'] == 'Implicit':
			data.append(obj)
		else:
			size += 1
		total += 1

	print('Total {0} data, implicit data {1}'.format(total, size))
	f.close()
	return data

if __name__ == '__main__':
	#read_data('train_pdtb.json')
	data = read_data('dev_pdtb.json')
	n = 1
	for k in data[n]:
		print(k, data[n][k])
