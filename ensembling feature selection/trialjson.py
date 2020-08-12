import json
path_dict = {"path":'kgfkfwejkfbwejfbwi'}
outfile = open('datastore.json','w')
json.dump(path_dict,outfile)
outfile.close()

infile = open('datastore.json','r')
path_dict = json.load(infile)
infile.close()
path_dict.update({'reorder':'jygdfweb'})
#print(path_dict)
path_dict.update({'reorder':'sgjsys'})
#print(path_dict)

outfile = open('datastore.json','w')
json.dump(path_dict,outfile)
outfile.close()

infile = open('datastore.json','r')
path_dict = json.load(infile)
print(path_dict['reorder'])
infile.close()