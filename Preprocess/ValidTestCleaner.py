# -*- coding: utf-8 -*-

"""
Created on Tue Oct 31 19:31:49 2017

@author: eduardofierro

First step of pre-processing data: 

- Clean and the provided dev/test data. 

"""

import re # Regular Expressions
import pickle

############### FILE PARAMS ###############

data_dir = "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/WIT3_en-fr/"
out_dir = "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Preprocess/"

############### FILE PARAMS ###############

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

def import_attributes(text, word):
    
    '''
    Function to import attributes from TED talks EN-FR - VALID AND TEST. 
    
    Params: 
    @text: Text imported using import_raw_text()
    @word: Attribures to search: Either 'url', 'description', 'keywords', 'talkid' or 'title'. 
           
    Returns: 
    @keywords: List of attributes per talk, identified by order. 
    '''
    
    attribute = '<' + word + '>'
    end_attribute = '</' + word + '>'
    
    activation = False
    mask = [line[0:len(attribute)] for line in text]
    mask = [True if line == attribute else False for line in mask]
    
    if (sum(mask) == 0):
        
        # A patch to previous functions. 
        # This text files come different by origin
        spaced_attribute = '    ' + attribute
        mask = [line[0:len(spaced_attribute)] for line in text]
        mask = [True if line == spaced_attribute else False for line in mask]
        activation = True
    
    return_list = []
    for x in range(0, len(mask)): 
        
        if mask[x] == True:
            
            clean_attr = re.sub('\n', '', text[x])
            clean_attr = re.sub(end_attribute, '', clean_attr)
            if activation == False:
                clean_attr = re.sub(attribute, '', clean_attr)
            else: 
                clean_attr = re.sub(spaced_attribute, '', clean_attr)
            return_list.append(clean_attr)
        
    return return_list 
    
def build_corpora(raw_text): 
    
    '''
    Function that receives raw text, and cleans corpus. 
    
    Params: 
    @raw_text: Raw text, imported using import_raw_text()
    
    Returns: 
    @complete_corpora: Clean text from corpus
    
    '''    
    complete_corpora = []
    for i, line in enumerate(raw_text):
        
        if line[0:4] == "<seg":
            
            text = re.sub(' </seg>', '', line)
            text = re.sub('<seg id="[0-9]*"> ', '', text)
            complete_corpora.extend(text) 
    
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
    
    print("Building dev and test sets...")
    filename_dic = {"IWSLT16.TED.dev2010.en-fr.en.xml" : "dev.en", 
                   "IWSLT16.TED.dev2010.en-fr.fr.xml" : "dev.fr", 
                   "IWSLT16.TED.tst2010.en-fr.en.xml" : "tst2010.en", 
                   "IWSLT16.TED.tst2010.en-fr.fr.xml" : "tst2010.fr",   
                   "IWSLT16.TED.tst2011.en-fr.en.xml" : "tst2011.en",  
                   "IWSLT16.TED.tst2011.en-fr.fr.xml" : "tst2011.fr",   
                   "IWSLT16.TED.tst2012.en-fr.en.xml" : "tst2012.en", 
                   "IWSLT16.TED.tst2012.en-fr.fr.xml" : "tst2012.fr",   
                   "IWSLT16.TED.tst2013.en-fr.en.xml" : "tst2013.en", 
                   "IWSLT16.TED.tst2013.en-fr.fr.xml" : "tst2013.fr",  
                   "IWSLT16.TED.tst2014.en-fr.en.xml" : "tst2014.en", 
                   "IWSLT16.TED.tst2014.en-fr.fr.xml" : "tst2014.fr"} 
    
    print("Saving metadada as pickles in: \n" + out_dir)
    for file in filename_dic.keys():
        
        file_text = import_raw_text(data_dir + file)
        
        urls = import_attributes(file_text, "url")
        speakers = import_attributes(file_text, "title")
        keywords = import_attributes(file_text, 'keywords')
        talkid = import_attributes(file_text, 'talkid') 
        medatada = [speakers, keywords, urls, talkid]
        pickle.dump(medatada, open( out_dir + filename_dic[file] + "_metadata.p", "wb" ) )    

        clean_text = build_corpora(file_text)        
        export_corpus(out_dir, filename_dic[file], clean_text)         

           
    print("READY!!!")
    
if __name__ == '__main__':
    main()