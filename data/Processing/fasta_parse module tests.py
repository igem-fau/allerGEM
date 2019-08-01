#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fasta_parse


# In[2]:


help(fasta_parse)


# In[3]:


test = fasta_parse.FASTA("A2BIM8.1.fasta")
print(test.header)
print('\n')
print(test.aa_seq)
print(test.comments) # None in this file.
print(len(test))
one_hot = test.one_hot()
print(one_hot)

