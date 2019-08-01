#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup
from Bio import Entrez, SeqIO

Entrez.email = "florianwolz@gmail.com"

overview_url = 'http://allergen.org/treeview.php'
search_url = 'http://allergen.org/{}'

overview = urllib.request.urlopen(overview_url)
overview_soup = BeautifulSoup(overview, 'html.parser')


# In[2]:


allergen_tree = overview_soup.find('ul', id="allergentree")


# In[3]:


search_urls = [search_url.format(category.find("cite").find("a")["href"]).replace(' ', '%20') for category in allergen_tree.children]


# In[4]:


the_links = []
for url in search_urls:
    the_list = urllib.request.urlopen(url)
    soup = BeautifulSoup(the_list, 'html.parser')
    
    # Get the links for all the elements
    links = [a["href"] for a in soup.findAll('a', href=re.compile('viewallergen.php\?aid=*'))]
    
    the_links = the_links + links


# In[6]:


fasta = []
file = open("../data/allergen_org_unfiltered.fasta", "w")

for link in the_links:
    the__link = search_url.format(link)
    
    try:
        the_page = urllib.request.urlopen(search_url.format(link))
    except:
        continue
    soup = BeautifulSoup(the_page, 'html.parser')
    
    table = soup.findAll('table')[-1]
    links = [td.find('a') for td in table.findAll('td')[2:4] if td.find('a')]
    if len(links) == 0: continue
        
    current = 0
    successful = False
    while True:
        if current >= len(links): break
            
        link = links[current]["href"]
        fasta = None
    
        try:
            if "ncbi.nlm" in link:
                name = link.replace('http://www.ncbi.nlm.nih.gov/nuccore/', '')
                handle = Entrez.efetch(db = "protein", id = name, rettype="fasta", retmode="text")
                fasta = handle.read()
            else:
                r = urllib.request.urlopen(link + ".fasta")
                fasta = r.read().decode('utf-8')
        except Exception as e:
            print(e)
            current += 1
            continue
            
        successful = True
        
        # Write the file
        if len(fasta) > 0:
            if fasta[-1] != '\n': fasta += '\n'
            file.write(fasta)
        
        # Done
        break
    
    # Print all links that failed
    if not successful: print("{} {}".format(the__link, link))

file.close()
print("Finished.")

