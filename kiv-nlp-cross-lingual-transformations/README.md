# Cross-Lingual Methods for transformation

The goal is to find a matrix **T** and then transform space **X** using the
matrix **T** as follows:

![transformation](../img/transformation.jpg)


## Data 


#### Analogies data
copy into folder 
https://github.com/brychcin/cross-ling-analogies
    ```
    ./data/analogies
    ```
#### Fasttext Data 
Download Czech and English fasttext embeddings

https://fasttext.cc/docs/en/crawl-vectors.html

copy into 
```
 ./data/emb-test
 ```


## Setup

1) #### Clone github repository 
   [The repository](https://github.com/pauli31/cross-lingual-transform-method.git)
   ```
   git clone https://github.com/pauli31/cross-lingual-transform-method.git
   ```

2) #### Setup conda
    Check version
    ```
    # print version
    conda -V
   
    # print available enviroments
    conda info --envs
    ```
    Create conda enviroment
   
    ```
    # create enviroment 
    conda create --name cross-lingual-transformation-sentiment python=3.7 -y
    
    # check that it was created
    conda info --envs
   
    # activate enviroment
    conda activate cross-lingual-transformation-method
   
    # see already installed packages
    pip freeze  
    ```
   
   Install requirements
   ```
   pip install -r requirements.txt
   ```
   
3) #### Test it
    ```
   cd kiv-nlp-cross-lingual-transformations
   python3 ./src/transformations/CanonicalCorrelationAnalysis.py 
   python3 ./src/transformations/LeastSquareTransformation.py
   python3 ./src/transformations/OrthogonalRankingTransformation.py
   python3 ./src/transformations/OrthogonalTransformation.py
   python3 ./src/transformations/RankingTransformation.py
   python3 ./src/transformations/Analogies.py
   ```

4) #### Setup transformation library for usage
   Go to the library
   ```
   cd  kiv-nlp-cross-lingual-transformations
   pip install .
   # or
   pip install --upgrade .
   # or
   pip install -v --upgrade .
   ```

5) #### Import the transformation
   ```
   from transformations import CanonicalCorrelationAnalysis
   cca = CanonicalCorrelationAnalysis(method='torch')
   ```


## License
The dataset and code can be freely used for academic and research purposes. It is strictly prohibited to use the dataset for any commercial purpose.

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Publication:
--------

If you use our dataset, software or approach for academic research, please cite the our [paper](https://arxiv.org/abs/2204.13915).

```
@InProceedings{priban-tsd-2022,
    author="P{\v{r}}ib{\'a}{\v{n}}, Pavel
    and {\v{S}}m{\'i}d, Jakub
    and Mi{\v{s}}tera, Adam
    and Kr{\'a}l, Pavel",
    editor="Sojka, Petr
    and Hor{\'a}k, Ale{\v{s}}
    and Kope{\v{c}}ek, Ivan
    and Pala, Karel",
    title="Linear Transformations for Cross-lingual Sentiment Analysis",
    booktitle="Text, Speech, and Dialogue",
    year="2022",
    publisher="Springer International Publishing",
    address="Cham",
    pages="125--137",
    isbn="978-3-031-16270-1"
}
```


Contributors:
--------

Pavel Přibáň,
Jakub Šmíd,
Adam Mištera

Contact:
--------
pribanp@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)
