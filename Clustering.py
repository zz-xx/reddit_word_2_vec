import nltk.data
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np;import os
import re
import logging
import sqlite3
import time
import sys
import multiprocessing
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import cycle



class Clustering:


    def __init__(self, noOfClusters, noOfComments, noOfTopWords ):
        
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
        ])

        self.ENGLISH_STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
                      "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                      "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                      "itself", "they", "them", "their", "theirs", "themselves", "what", 
                      "which", "who", "whom", "this", "that", "these", "those", "am", "is", 
                      "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                      "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", 
                      "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                      "with", "about", "against", "between", "into", "through", "during", "before", 
                      "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", 
                      "off", "over", "under", "again", "further", "then", "once", "here", "there", 
                      "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
                      "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
                      "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", 
                      "now" 
                      ]
        self.EXPLICIT_WORDS = ["anal", "anus", "ballsack", "blowjob", "butt", "blow", "job", "boner", "clitoris", 
                          "cock", "cunt", "dick", "dildo", "dyke", "fag", "fuck", "fuckin", "jizz", "labia", "muff", 
                          "nigger", "nigga", "penis", "piss", "pussy", "scrotum", "sex", "shit", "slut", 
                          "smegma", "spunk", "twat", "vagina", "wank", "whore"
                        ]

        self.noOfClusters = noOfClusters 
        self.noOfComments = noOfComments
        self.noOfTopWords = noOfTopWords + 10


    def clean_text(self, all_comments, out_name):
        
        out_file = open(out_name, 'w')
        
        for pos in range(len(all_comments)):
        
            #get the comment
            val = all_comments.iloc[pos]['body']
            
            #normalize tabs and remove newlines
            no_tabs = str(val).replace('\t', ' ').replace('\n', '')
            
            #remove all characters except A-Z and a dot.
            alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs)
            
            #normalize spaces to 1
            multi_spaces = re.sub(" +", " ", alphas_only)
            
            #strip trailing and leading spaces
            no_spaces = multi_spaces.strip()
            
            #normalize all charachters to lowercase
            clean_text = no_spaces.lower()
            
            #get sentences from the tokenizer, remove the dot in each.
            sentences = self.tokenizer.tokenize(clean_text)
            sentences = [re.sub("[\.]", "", sentence) for sentence in sentences]
            
            #if the text has more than one space (removing single word comments) and one character, write it to the file.
            if len(clean_text) > 0 and clean_text.count(' ') > 0:
                for sentence in sentences:
                    out_file.write("%s\n" % sentence)
                    #print(sentence)

        out_file.close()


    def clustering_on_wordvecs(self, word_vectors, num_clusters):

        #initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++')
        idx = kmeans_clustering.fit_predict(word_vectors)
        
        return kmeans_clustering.cluster_centers_, idx


    def get_top_words(self, index2word, k, centers, wordvecs):
        
        tree = KDTree(wordvecs)

        #closest points for each Cluster center is used to query the closest 20 points to it.
        closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]
        closest_words_idxs = [x[1] for x in closest_points]

        #word Index is queried for each position in the above array, and added to a Dictionary.
        closest_words = {}
        for i in range(0, len(closest_words_idxs)):
            closest_words['Cluster #' + str(i+1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]

        #DataFrame is generated from the dictionary.
        df = pd.DataFrame(closest_words)
        df.index = df.index+1

        return df


    def display_cloud(self, cluster_num, cmap):

        wc = WordCloud(background_color="gray", max_words=2000, max_font_size=80, colormap=cmap)
        #wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num).zfill(2)] if word not in ENGLISH_STOP_WORDS]))
        
        try:
            words = [word for word in self.top_words['Cluster #' + str(cluster_num).zfill(2)] if word not in self.ENGLISH_STOP_WORDS and len(word) > 2]
            wordcloud = wc.generate(' '.join([word for word in words if not any(explicitWord in word for explicitWord in self.EXPLICIT_WORDS)]))                                                                                                #if any(xs in s for xs in matchers)                        
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig('static\\images\\' + 'img_' + str(cluster_num), bbox_inches='tight')
        
        except Exception:
            pass


    def print_word_table(self, table, key):
        return pd.DataFrame(table, columns=[key, 'similarity'])


    def main_func(self):

        #------------read and load data carefully from sqlite db --------------------

        sql_con = sqlite3.connect("T:\\2018++\\BE\LP\\LP2\\extras\\database.sqlite\\database.sqlite")
        cursor = sql_con.cursor()
        print("Connected to database")
        cursor.execute("SELECT body FROM May2015")

        count = 0
        col_names = ['body']
        sql_data  = pd.DataFrame(columns = col_names)
        #print(sql_data)
        #print(len(sql_data))

        start = time.time()

        for row in cursor:
            
            if count == self.noOfComments:
                break
            
            temp_dic = {'body':row[0]}
            sql_data.loc[len(sql_data)] = temp_dic

            count+=1


        print(len(sql_data))
        print('Total time: ' + str((time.time() - start)) + ' secs')


        start = time.time()
        self.clean_text(sql_data, 'out_full')
        print('Total time: ' + str((time.time() - start)) + ' secs')


        #--------- training and saving model ------------------------

        start = time.time()

        #dimensionality of the hidden layer representation
        num_features = 100 
        #minimum word count to keep a word in the vocabulary
        min_word_count = 40
        #number of threads to run in parallel 
        #set to total number of cpus.  
        num_workers = multiprocessing.cpu_count()
        #context window size (on each side)        
        context = 5
        #downsample setting for frequent words                                                              
        downsampling = 1e-3   

        #initialize and train the model. 
        print("Training model...")
        model = word2vec.Word2Vec(LineSentence('out_full'), workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

        model.init_sims(replace=True)

        #save the model
        model_name = "model_full_reddit"
        model.save(model_name)

        print('Total time: ' + str((time.time() - start)) + ' secs')


        Z = model.wv.syn0
        print(Z[0].shape)
        #print(Z[0])


        #---------cluster the word vectors obtained-------------
        start = time.time()
        centers, clusters = self.clustering_on_wordvecs(Z, self.noOfClusters)
        print('Total time: ' + str((time.time() - start)) + ' secs')


        start = time.time()
        centroid_map = dict(zip(model.wv.index2word, clusters))
        print('Total time: ' + str((time.time() - start)) + ' secs')


        #------just for display------
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        self.top_words = self.get_top_words(model.wv.index2word, self.noOfTopWords, centers, Z)

        print(self.top_words)


        #-----------make word cloud for each cluster--------
        for i in range(self.noOfClusters):
            col = next(self.cmaps)
            self.display_cloud(i+1, col)

        #print dataframe
        #print(self.print_word_table(model.wv.most_similar_cosmul(positive=['big', 'small'], negative=['high']), 'Analogy'))

        return True

#obj = Clustering(20, 10000, 10)
#obj.main_func()
#20, 25000, 10