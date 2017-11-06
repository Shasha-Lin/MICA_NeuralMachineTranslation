# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:17:44 2017

@author: eduardofierro

Function to merge any two text files and create a new file with merged lines: 

"""

import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, default="./en.txt", help='File 1 - Dir + Filename')
parser.add_argument('--file2', type=str, default="./fr.txt", help='File 2 - Dir + Filename')
parser.add_argument('--outdir', type=str, default="./", help='Output directory for concatenated file')
parser.add_argument('--lang1', type=str, default="en", help='Language 1 id. i.e. en=english')
parser.add_argument('--lang2', type=str, default="fr", help='Language 2 id. i.e. fr=french')
parser.add_argument('--term', type=str, default=".bpe2bpe", help='file ending for output fule')
opt = parser.parse_args()
print(opt)

def concatanate_by_tabulation(str1, str2):
    
    if re.search("\t",str1):
        raise ValueError("Tabulation found in string 1")
        
    if re.search("\t",str2): 
        raise ValueError("Tabulation found in string 2")
     
    str1 = re.sub("\n", "", str1)
    str2 = re.sub("\n", "", str2)
    return(str1 + "\t" + str2)

def file_merger(file1, file2, outdir, lang1, lang2, term=".txt"):
    
    '''
    Merges two list of strings together (concatenate by elemnt) and separate by tab. 
    
    Params: 
    @file1 = Dir + filename of strins file one (input)
    @file1 = Dir + filename of strins file two (target)
    @outdir = Directory where file is going to be located
    @lang1 = Language 1 (i.e. "en")
    @lang2 = Language 2 (i.e. "fr")
    @term =  file ending. Default = ".txt".        
    '''
    
    
    file1_lines = []
    with open(file1) as f:
        for i, line in enumerate(f): 
            
            file1_lines.append(line)   
    f.close
    file2_lines = []
    with open(file2) as f:
        for i, line in enumerate(f): 
            
            file2_lines.append(line)       
    f.close
    
    return_list = [concatanate_by_tabulation(file1_lines[x], file2_lines[x]) for x in range(0, len(file1_lines))]
    
    file_name = outdir + "/" + lang1 + "-" + lang2 + term
    
    out_file = open(file_name, 'w')
    
    for line in return_list:
            out_file.write(line + "\n") 
    out_file.close    
    print("File exported: " + file_name)
    
            
if __name__ == '__main__':
    print("\n")
    file_merger(opt.file1, opt.file2, opt.outdir, opt.lang1, opt.lang2, opt.term) 