#!/usr/bin/env python
# coding: utf-8

# In[14]:


from flask import Flask, request, render_template


# In[15]:


from transformers import pipeline


# In[16]:


classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")


# In[17]:


app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        input_txt = request.form.get("input_txt")
        print(input_txt)
        r = classifier(input_txt)
        return(render_template("index.html", result = r))
    else:
        return(render_template("index.html", result = "Waiting"))


# In[18]:


if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




