from flask import Flask, render_template,request


import pandas as pd
customer = pd. read_csv("E:/MY_DS/newfood.csv")
customer. shape
customer. Breakfast
customer. isna(). sum()
from sklearn. feature_extraction. text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(customer. Breakfast)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #174, 113

# calculating the dot product using sklearn's linear_kernel()
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# creating a mapping of anime name to index number 
customer_index = pd.Series(customer.index, index = customer['Breakfast'])
customer_index
customer_id = customer_index["Mysore Bonda, Poha, Hotdog Sandwich"]
customer_id

def get_recommendations(Breakfast, topN):    
    # Getting the food index using its title 
    customer_id = customer_index[Breakfast]
    topN = 10
    # Getting the pair wise similarity score for all the anime's with that 
    cosine_scores = list(enumerate(cosine_sim_matrix[customer_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True) 
    cosine_scores_N = cosine_scores[0: topN+1]
     
    customer_idx  =  [i[0] for i in cosine_scores_N]
    customer_scores =  [i[1] for i in cosine_scores_N]
    customer_similar_show = pd.DataFrame(columns=["Breakfast", "score"])
    customer_similar_show["Breakfast"] = customer.loc[customer_idx, "Breakfast"]
    customer_similar_show["score"] = customer_scores
    customer_similar_show.reset_index(inplace = True)  
    #print (customer_similar_show)
    del customer_similar_show['index']    
    return customer_similar_show
get_recommendations('Dosa, Mysore Bonda', topN =10)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about',methods=['POST'])
def about():
    breakfastname = request.form["breakfastname"]
    get_recommendations(breakfastname,topN =10)
    df = get_recommendations(breakfastname,topN =10)
    return render_template('result.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)


    
    