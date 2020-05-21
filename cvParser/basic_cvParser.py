import os
import sys
import spacy
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
#import en_core_web_sm
from io import StringIO
from collections import Counter
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

##function that pdf file into text
def pdfextract(file):
    count = 0
    text = []
    try:
        fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
        countpage = fileReader.getNumPages()
        while count < countpage:    
            pageObj = fileReader.getPage(count)
            count += 1
            t = pageObj.extractText()
            #print (t)
            text.append(t)
    except Exception as e:
        return '0'
    return text


#function that does phrase matching and builds a candidate profile
def create_profile(file):
    text = pdfextract(file)
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #print(text)

    #below is the csv where we have all the keywords, you can customize your own
    try:
        keyword_dict = pd.read_csv(sys.argv[2])
    except Exception as e:
        print(e)
        exit()

    NLP_words = [nlp(text.lower()) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ML_words = [nlp(text.lower()) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    DL_words = [nlp(text.lower()) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    python_words = [nlp(text.lower()) for text in keyword_dict['Language'].dropna(axis = 0)]
    protocol_words = [nlp(text.lower()) for text in keyword_dict['Protocols'].dropna(axis = 0)]

    #print(NLP_words, '\n', ML_words, "\n", DL_words, "\n", python_words, "\n", protocol_words)

    ##Creating object of matcher library to add skills that company wants
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('Language', None, *python_words)
    matcher.add('Protocols', None, *protocol_words)
    doc = nlp(text)

    #print(type(doc), type(text))

    ##Check the resume for skills which company wants
    d = []  
    matches = matcher(doc)
    #print(matches, type(matches), type(matcher), type(doc))
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    #print(keywords)
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
       
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

    return dataf
        
        
def main(argv):
    ##Function to read resumes from the folder one by one
    try:
        mypath = sys.argv[1] #enter your path here where you saved the resumes
        onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        #print(onlyfiles)

        final_database=pd.DataFrame()
        i = 0 
        while i < len(onlyfiles):
            file = onlyfiles[i]
            dat = create_profile(file)
            final_database = final_database.append(dat)
            i +=1
        #print(final_database)
    except Exception as e:
        print(e)

    #code to count words under each category and visulaize it through Matplotlib
    final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
    final_database2.reset_index(inplace = True)
    final_database2.fillna(0, inplace=True)
    new_data = final_database2.iloc[:, 1:]
    new_data.index = final_database2['Candidate Name']

    print(new_data)

    #execute the below line if you want to see the candidate profile in a csv format
    sample2=new_data.to_csv('sample.csv')

    plt.rcParams.update({'font.size': 10})
    ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
    labels = []
    for j in new_data.columns:
        for i in new_data.index:
            label = str(j)+": " + str(new_data.loc[i][j])
            labels.append(label)
    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width/2., y + height/2., label, ha='center', va='center')
    plt.show()

##code to execute/call the above functions
if __name__ == '__main__' :
    #print(sys.argv)
    if len(sys.argv) >= 3:
        main(sys.argv[1:])
    else:
        print("Usage: python3 basic_cvParser.py Path_of_resumeFolder Path/csvFile.csv")
