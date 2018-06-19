import pandas as pd, numpy as np, csv
import math, json, sys

# Filepaths
DATASET = "./../data/comments.json"

# Output Files
USER_COMMENTS = "./train_balanced_user.csv"


# Load the dataset
sys.stdout.write("Loading dataset ..."+'\r')
sys.stdout.flush()
json_data = json.load(open(DATASET,"r"))

# Set of unique users
sys.stdout.write("Calculating set of users..."+'\r')
sys.stdout.flush()
data = []
for k, v in json_data.iteritems():
	data.append([k,v['author'],v['text']])
users = [row[1] for row in data]
users = set(users)


output_file = open(USER_COMMENTS,'w')
wr = csv.writer(output_file, quoting=csv.QUOTE_ALL)


# Accumulate comments of each user into paragraphs
sys.stdout.write("Accumulating user comments..."+'\r')
sys.stdout.flush()
for ind, user in enumerate(users):
	if ind%100 ==0:
		sys.stdout.write(str(ind+1) +"/"+ str(len(users))+" users done..."+'\r' )
		sys.stdout.flush()

	comments = [row[2] for row in data if row[1]==user]
	comments = [x for x in comments if str(x) != 'nan']
	comment = " <END> ".join(comments)


	ls=[]
	ls.append(user)
	ls.append(comment)
	wr.writerow(ls)
