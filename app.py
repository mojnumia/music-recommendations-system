from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
import numpy as np
import librosa as lr
from sklearn.metrics.pairwise import cosine_similarity
df= pd.read_json('sample.json')
dat= pd.read_csv('featuredataf.csv')

app= Flask(__name__)



@app.route('/')
def test():
	return render_template('home.html')

@app.route('/recommend', methods=['POST','GET'])
def recommend():
	if request.method == 'POST':
		#grab data by post method
		data= request.form['a']
		#load music file path
		signal,sr= lr.load(data)
		# extracting music feature mfccs
		mfcc= lr.feature.mfcc(signal,sr=sr,n_mfcc=13)
		# tranverse the matrix
		mfcc=mfcc.T
		#convert into list
		mfccs= mfcc.tolist()[-1]
		mfccs= [mfccs]
		#vectorize the mfcc fature from dataframe
		vector=[]
		for v in df['mfcc']:
			vector.append(v)
		# covert vector into numpy array
		vector_array= np.array(vector)
		#find similarity
		similarity= cosine_similarity(mfccs,vector_array)
		#convert similarity into numpy array
		sort_array= np.array(similarity)

		# sorting similarity low to high value
		sort_similarity= np.argsort(sort_array)
		# find top 4 similarity
		simi_data= pd.DataFrame(sort_similarity.T)
		max_simi= simi_data.tail(4)
		# convert array and then list
		max_simi=np.array(max_simi)
		max_simi=max_simi.tolist()
		flat_max_simi= sum(max_simi,[])
		flat_max_simi.reverse()
		# create a dummy dictionary for data save
		dic={
			"genre": [],
			"filename": [],
			"filepath": []
		}
		#saving the recommended data
		for dataa in flat_max_simi:
			h = dat.loc[dat['Unnamed: 0'] == dataa, 'genre'].item()
			i = dat.loc[dat['Unnamed: 0'] == dataa, 'filename'].item()
			j = dat.loc[dat['Unnamed: 0'] == dataa, 'filepath'].item()
			dic["genre"].append(h)
			dic["filename"].append(i)

			dic["filepath"].append(j)
		data_dic= pd.DataFrame(dic)
		final_data= data_dic.T.to_dict()


		return render_template('recommend.html',items= data,list=final_data)



if __name__ == "__main__":
	app.run(debug=True)
