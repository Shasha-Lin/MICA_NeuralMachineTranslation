# -*- coding: utf-8 -*-

"""
Created on Tue Oct 31 14:05:49 2017

@author: eduardofierro

First step of pre-processing data: 

- Clean and split into validation/Train set the main train. 


"""

import numpy as np
import re # Regular Expressions
import pickle

############### FILE PARAMS ###############

data_dir = "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/WIT3_en-fr/"
out_dir = "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/"
train_split = 80
valid_split = 20

############### FUNCTIONS TO IMPORT METADADA FROM MAIN FILE ###############

def import_raw_text(path):
        
    '''
    Function to import VERY raw text. 
    
    Params: 
    @path: Path of the name of the file; pressumably train.tags.en-fr.en / train.tags.en-fr.fr. 
           Data downloded from: https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=en&tlang=fr
           
    Return: 
    @text: Raw text
    '''
    
    text = []
    with open(path) as f:
        for i, line in enumerate(f): 
            
            text.append(line)   
            
    return text

def import_urls(text):
    
    '''
    Function to import URLs from TED talks EN-FR data. 
    
    Params: 
    @text: Text imported using import_raw_text()
           
    Returns: 
    @urls: List of urls, identified by order. 
    '''

    mask = [line[0:5] for line in text]
    mask = [True if line == '<url>' else False for line in mask]
    
    urls = []
    for x in range(0, len(mask)): 
        if mask[x] == True:
            clean_url = re.sub('\n', '', text[x])
            clean_url = re.sub('<url>', '', clean_url)
            clean_url = re.sub('</url>', '', clean_url)
            urls.append(clean_url)
        
    return urls 

def import_keywords(text):
    
    '''
    Function to import keywords from TED talks EN-FR data. 
    
    Params: 
    @text: Text imported using import_raw_text()
           
    Returns: 
    @keywords: List of keywords, identified by order. 
    '''
    
    mask = [line[0:10] for line in text]
    mask = [True if line == '<keywords>' else False for line in mask]
    
    keywords = []
    for x in range(0, len(mask)): 
        if mask[x] == True:
            clean_keywords = re.sub('\n', '', text[x])
            clean_keywords = re.sub('<keywords>', '', clean_keywords)
            clean_keywords = re.sub('</keywords>', '', clean_keywords)
            keywords.append(clean_keywords)
        
    return keywords 

def import_speaker(text):
    
    '''
    Function to import speakers from TED talks EN-FR data. 
    
    Params: 
    @text: Text imported using import_raw_text()
           
    Returns: 
    @speaker: List of speakers, identified by order. 
    '''
    
    mask = [line[0:9] for line in text]
    mask = [True if line == '<speaker>' else False for line in mask]
    
    speaker = []
    for x in range(0, len(mask)): 
        if mask[x] == True:
            clean_speaker = re.sub('\n', '', text[x])
            clean_speaker = re.sub('<speaker>', '', clean_speaker)
            clean_speaker = re.sub('</speaker>', '', clean_speaker)
            speaker.append(clean_speaker)
        
    return speaker 

def import_talkid(text):
    
    '''
    Function to import talkid from TED talks EN-FR data. 
    
    Params: 
    @text: Text imported using import_raw_text()
           
    Returns: 
    @talkid: List of talkid, identified by order. 
    '''
    
    mask = [line[0:8] for line in text]
    mask = [True if line == '<talkid>' else False for line in mask]
    
    talkid = []
    for x in range(0, len(mask)): 
        if mask[x] == True:
            clean_talkid = re.sub('\n', '', text[x])
            clean_talkid = re.sub('<talkid>', '', clean_talkid)
            clean_talkid = re.sub('</talkid>', '', clean_talkid)
            talkid.append(clean_talkid)
        
    return talkid 

############### FUNCTIONS TO CREATE TRAIN AND VALIDATION SET ###############
   
def id_splitter(id_List, per_train, per_valid, random_seed=1234):
    
    '''
    Function to split an id_List into Train, Valid and Test. 
    
    Params: 
    @id_List - List of ids
    @per_train - Percentage on train, in format 0 to 100
    @per_valid - Percentage on validation, in format 0 to 100
    @random_seed - Numpys random seed. Defaul 1234. 
    
    Return: 
    @train, @valid, @test - id_List randomly splitted into this 3 sets.     
    '''
    
    if per_train < 0: 
        raise ValueError('per_train out of bound: Selected {} but need a number between 0 and 100'.format(per_train))
     
    if per_train > 100: 
        raise ValueError('per_train out of bound: Selected {} but need a number between 0 and 100'.format(per_train))
        
    if per_valid < 0: 
        raise ValueError('per_valid out of bound: Selected {} but need a number between 0 and 100'.format(per_valid))
        
    if per_valid  > 100: 
        raise ValueError('per_valid out of bound: Selected {} but need a number between 0 and 100'.format(per_valid))
        
    if per_train + per_valid > 100: 
        raise ValueError('per_valid and per_train add more than 100.')
        
    np.random.seed(random_seed)    
    random = np.random.randint(0, 100, len(id_List))
    
    train = np.array(id_List)[random < per_train]
    valid = np.array(id_List)[(random >= (per_train)) & (random < (per_train + per_valid))]
    
    if (per_train + per_valid) < 100:
        test = np.array(id_List)[random >= (per_train + per_valid)]
        return train.tolist(), valid.tolist(), test.tolist() 
        
    else: 
        return train.tolist(), valid.tolist(), None   
        
def search_id(my_id, raw_text):
    
    ''' 
    Searches for any given ID on original file. 
    Returns index on original file where the id is found
     
    Params: 
    @my_id: STRING!!! id on the id list. 
    @raw_text: Raw unprocessed text. 
    
    Returns: 
    @line_id: Index on original 'raw_text' where the id was found.      
    '''
    
    line2search = "<talkid>{}</talkid>\n".format(my_id)
    line_id = [line == line2search for line in raw_text].index(True)
        
    return line_id

def raw_line_search(my_id, raw_text):
    
    '''
    Function to search text within the raw text based on talkid
    
    Params:
    @my_id: STRING! Any given ID. 
    @raw_text: Text to search, eighter english or frech, imported
               with import_raw_text()
               
    returns: 
    @return_text: Clean text of the given ID. 
    
    '''
    
    begin_index = search_id(my_id, raw_text) + 3

    end_index = begin_index
    text_stop = False
    while text_stop is False: 
        
        if (raw_text[end_index][0] == "<"):
            text_stop = True
        
        else: 
            end_index += 1
      
    return_text = raw_text[begin_index:end_index]
    
    return return_text  

def build_corpora(index_list, raw_text): 
    
    '''
    Function that receives an Index List, and cleans corpus. 
    Returns clean text in english and frensh, respectively.
    
    Params:
    @index_list: List of the indexes (IN STRING FORMAT!) of ids 
    @raw_text: Text to search, eighter english or frech, imported
               with import_raw_text()
               
    returns: 
    @return_text: Clean text of the given ID.     
    
    '''
    
    complete_corpora = []
    for i, index_val in enumerate(index_list):
        
        text = raw_line_search(index_val, raw_text)
        complete_corpora.extend(text) 
        
        if i % 200 ==0:
            percent = round(i*100/len(index_list), 2)
            print("Building Corpus: {}%".format(percent))
    
    print("Corpus READY!!!")
    return complete_corpora  
    
    
def export_corpus(save_dir, file_name, list_export): 
    
    '''
    Function that writes as text a list of lines. 
    
    Params:
    @save_dir: Directory to save
    @file_name: Name of file to save
    @list_export: list of lines of text
               
    returns: 
    NONE 
    
    '''
    
    file = open(save_dir + file_name, 'w')
    for line in list_export: 
        file.write(line)
    file.close
    print("File {} written in: \n {}".format(file_name, save_dir))
    
    
def main():
    
    text_en = import_raw_text(data_dir + "train.tags.en-fr.en")
    text_fr = import_raw_text(data_dir + "train.tags.en-fr.fr")
    speakers = import_speaker(text_en)
    keywords = import_keywords(text_en)
    talkid = import_talkid(text_en)
    urls = import_urls(text_en)   
       
    train_ids, valid_ids, test_ids = id_splitter(talkid, train_split, valid_split)
    
    print("Building Train Coprpus - English")
    train_en = build_corpora(train_ids, text_en)    
    print("Building Train Coprpus - French")
    train_fr = build_corpora(train_ids, text_fr)
    
    if len(train_en) != len(train_en): 
        raise ValueError("Train English ({}) and French ({}) not same lengnth!!".format(train_en, train_fr))

    print("Building Dev Coprpus - English")
    valid_en = build_corpora(valid_ids, text_en)    
    print("Building Dev Coprpus - French")
    valid_fr = build_corpora(valid_ids, text_fr)

    if len(valid_en) != len(valid_fr): 
        raise ValueError("Valid English ({}) and French ({}) not same lengnth!!".format(train_en, train_fr))
       
    export_corpus(out_dir, "train.en", train_en)
    export_corpus(out_dir, "train.fr", train_fr)
    export_corpus(out_dir, "valid.en", valid_en)
    export_corpus(out_dir, "valid.fr", valid_fr)

    if (train_split + valid_split) < 100:
        print("Building Test Coprpus - English")
        test_en = build_corpora(test_ids, text_en)        
        print("Building Test Coprpus - French")
        test_fr = build_corpora(test_ids, text_fr)
        
        if len(test_en) != len(test_fr): 
            raise ValueError("Test English ({}) and French ({}) not same lengnth!!".format(train_en, train_fr))           
        
        export_corpus(out_dir, "test.en", test_en)
        export_corpus(out_dir, "test.fr", test_fr) 

    print("Saving metadada as pickle in: \n" + out_dir)
    medatada = [speakers, keywords, urls, talkid]
    pickle.dump(medatada, open( out_dir + "train_valid_metadata.p", "wb" ) )        
        
if __name__ == '__main__':
    main()