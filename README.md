Download Link: https://assignmentchef.com/product/solved-advanced-vector-space-models-assignment-4
<br>
In this assignment, we will examine some advanced uses of vector representations of words. We are going to look at two different problems:

<ol>

 <li>Solving word relation problems like analogies using word embeddings.</li>

 <li>Discovering the different senses of a “polysemous” word by clustering together its synonyms. You will use an open source Python package for creating and manipulating word vectors called <em>gensim.</em> Gensim lets you easily train word embedding models like word2vec.</li>

</ol>

<h1 id="part-1-exploring-analogies-and-other-word-pair-relationships">Part 1: Exploring Analogies and Other Word Pair Relationships</h1>

Word2vec is a very cool word embedding method that was developed by Thomas Mikolov and his collaborators. One of the noteworthy things about the method is that it can be used to solve word analogy problems like man is to king as woman is to [blank] The way that it works is to perform vector math. They take the vectors representing <em>king</em>, <em>man</em> and <em>woman</em> and perform some vector arithmetic to produce a vector that is close to the expected answer.<span id="MathJax-Element-1-Frame" class="MathJax" style="box-sizing: border-box; display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 14px; text-indent: 0px; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0px; min-height: 0px; border: 0px; padding: 0px; margin: 0px; position: relative;" tabindex="0" role="presentation" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><mi>k</mi><mi>i</mi><mi>n</mi><mi>g</mi><mo>&amp;#x2212;</mo><mi>m</mi><mi>a</mi><mi>n</mi><mo>+</mo><mi>w</mi><mi>o</mi><mi>m</mi><mi>a</mi><mi>n</mi><mo>&amp;#x2248;</mo><mi>q</mi><mi>u</mi><mi>e</mi><mi>e</mi><mi>n</mi></math>"><span id="MathJax-Span-1" class="math"><span id="MathJax-Span-2" class="mrow"><span id="MathJax-Span-3" class="mi">k</span><span id="MathJax-Span-4" class="mi">i</span><span id="MathJax-Span-5" class="mi">n</span><span id="MathJax-Span-6" class="mi">g</span><span id="MathJax-Span-7" class="mo">−</span><span id="MathJax-Span-8" class="mi">m</span><span id="MathJax-Span-9" class="mi">a</span><span id="MathJax-Span-10" class="mi">n</span><span id="MathJax-Span-11" class="mo">+</span><span id="MathJax-Span-12" class="mi">w</span><span id="MathJax-Span-13" class="mi">o</span><span id="MathJax-Span-14" class="mi">m</span><span id="MathJax-Span-15" class="mi">a</span><span id="MathJax-Span-16" class="mi">n</span><span id="MathJax-Span-17" class="mo">≈</span><span id="MathJax-Span-18" class="mi">q</span><span id="MathJax-Span-19" class="mi">u</span><span id="MathJax-Span-20" class="mi">e</span><span id="MathJax-Span-21" class="mi">e</span><span id="MathJax-Span-22" class="mi">n</span></span></span><span class="MJX_Assistive_MathML" role="presentation">king−man+woman≈queen</span></span> So We can find the nearest a vector in the vocabulary by looking for <span id="MathJax-Element-2-Frame" class="MathJax" style="box-sizing: border-box; display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 14px; text-indent: 0px; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0px; min-height: 0px; border: 0px; padding: 0px; margin: 0px; position: relative;" tabindex="0" role="presentation" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><mi>a</mi><mi>r</mi><mi>g</mi><mi>m</mi><mi>a</mi><mi>x</mi><mtext>&amp;#xA0;</mtext><mi>c</mi><mi>o</mi><mi>s</mi><mo stretchy=&quot;false&quot;>(</mo><mi>x</mi><mo>,</mo><mi>k</mi><mi>i</mi><mi>n</mi><mi>g</mi><mo>&amp;#x2212;</mo><mi>m</mi><mi>a</mi><mi>n</mi><mo>+</mo><mi>w</mi><mi>o</mi><mi>m</mi><mi>a</mi><mi>n</mi><mo stretchy=&quot;false&quot;>)</mo></math>"><span id="MathJax-Span-23" class="math"><span id="MathJax-Span-24" class="mrow"><span id="MathJax-Span-25" class="mi">a</span><span id="MathJax-Span-26" class="mi">r</span><span id="MathJax-Span-27" class="mi">g</span><span id="MathJax-Span-28" class="mi">m</span><span id="MathJax-Span-29" class="mi">a</span><span id="MathJax-Span-30" class="mi">x</span><span id="MathJax-Span-31" class="mtext"> </span><span id="MathJax-Span-32" class="mi">c</span><span id="MathJax-Span-33" class="mi">o</span><span id="MathJax-Span-34" class="mi">s</span><span id="MathJax-Span-35" class="mo">(</span><span id="MathJax-Span-36" class="mi">x</span><span id="MathJax-Span-37" class="mo">,</span><span id="MathJax-Span-38" class="mi">k</span><span id="MathJax-Span-39" class="mi">i</span><span id="MathJax-Span-40" class="mi">n</span><span id="MathJax-Span-41" class="mi">g</span><span id="MathJax-Span-42" class="mo">−</span><span id="MathJax-Span-43" class="mi">m</span><span id="MathJax-Span-44" class="mi">a</span><span id="MathJax-Span-45" class="mi">n</span><span id="MathJax-Span-46" class="mo">+</span><span id="MathJax-Span-47" class="mi">w</span><span id="MathJax-Span-48" class="mi">o</span><span id="MathJax-Span-49" class="mi">m</span><span id="MathJax-Span-50" class="mi">a</span><span id="MathJax-Span-51" class="mi">n</span><span id="MathJax-Span-52" class="mo">)</span></span></span><span class="MJX_Assistive_MathML" role="presentation">argmax cos(x,king−man+woman)</span></span>. Omar Levy has a nice explnation of the method in <a href="https://www.quora.com/How-does-Mikolovs-word-analogy-for-word-embedding-work-How-can-I-code-such-a-function">this Quora post</a>, and in his paper <a href="http://www.aclweb.org/anthology/W14-1618">Linguistic Regularities in Sparse and Explicit Word Representations</a>.

In addition to solving this sort of analogy problem, the same sort of vector arithmetic was used with word2vec embeddings to find relationships between pairs of words like the following:

<img decoding="async" alt="Examples of five types of semantic and nine types of syntactic questions in the Semantic- Syntactic Word Relationship test set" data-recalc-dims="1" data-src="https://i0.wp.com/computational-linguistics-class.org/assets/img/word2vec_word_pair_relationships.jpg?w=980" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/computational-linguistics-class.org/assets/img/word2vec_word_pair_relationships.jpg?w=980" alt="Examples of five types of semantic and nine types of syntactic questions in the Semantic- Syntactic Word Relationship test set" data-recalc-dims="1">

 </noscript>

In the first part of this homework, you will play around with the <a href="https://radimrehurek.com/gensim/index.html">gensim library</a> library. You will use <code class="highlighter-rouge">gensim</code> load a dense vector model trained using <code class="highlighter-rouge">word2vec</code>, and use it to manipulate and analyze the vectors.You can start by experimenting on your own, or reading through <a href="https://rare-technologies.com/word2vec-tutorial/">this tutorial on using word2vec with gensim</a>. You should familiarize yourself with the <a href="https://radimrehurek.com/gensim/models/keyedvectors.html">KeyedVectors documentation</a>.

The questions below are designed to familiarize you with the <code class="highlighter-rouge">gensim</code> Word2Vec package, and get you thinking about what type of semantic information word embeddings can encode. You’ll submit your answers to these questions when you submit your other homework materials.

Load the word vectors using the following Python commands:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">gensim.models</span> <span class="kn">import</span> <span class="n">KeyedVectors</span><span class="n">vecfile</span> <span class="o">=</span> <span class="s">'GoogleNews-vectors-negative300.bin'</span><span class="n">vecs</span> <span class="o">=</span> <span class="n">KeyedVectors</span><span class="o">.</span><span class="n">load_word2vec_format</span><span class="p">(</span><span class="n">vecfile</span><span class="p">,</span> <span class="n">binary</span><span class="o">=</span><span class="bp">True</span><span class="p">)</span></code></pre>

</figure>

<ul>

 <li>What is the dimensionality of these word embeddings? Provide an integer answer.</li>

 <li>What are the top-5 most similar words to <code class="highlighter-rouge">picnic</code> (not including <code class="highlighter-rouge">picnic</code> itself)? (Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.most_similar</code>)</li>

 <li>According to the word embeddings, which of these words is not like the others? <code class="highlighter-rouge">['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']</code> (Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.doesnt_match</code>)</li>

 <li>Solve the following analogy: “leg” is to “jump” as X is to “throw”. (Use the function <code class="highlighter-rouge">gensim.models.KeyedVectors.wv.most_similar</code> with <code class="highlighter-rouge">positive</code>and <code class="highlighter-rouge">negative</code> arguments.)</li>

</ul>

We have provided a file called <code class="highlighter-rouge">question1.txt</code> for you to submit answers to the questions above.

<h1 id="part-2-creating-word-sense-clusters">Part 2: Creating Word Sense Clusters</h1>

Many natural language processing (NLP) tasks require knowing the sense of polysemous words, which are words with multiple meanings. For example, the word <em>bug</em> can mean

<ol>

 <li>a creepy crawly thing</li>

 <li>an error in your computer code</li>

 <li>a virus or bacteria that makes you sick</li>

 <li>a listening device planted by the FBI</li>

</ol>

In past research my PhD students and I have looked into automatically deriving the different meaning of polysemous words like bug by clustering their paraphrases. We have developed a resource called <a href="http://paraphrase.org/">the paraphrase database (PPDB)</a> that contains of paraphrases for tens of millions words and phrases. For the target word <em>bug</em>, we have an unordered list of paraphrases including: <em>insect, glitch, beetle, error, microbe, wire, cockroach, malfunction, microphone, mosquito, virus, tracker, pest, informer, snitch, parasite, bacterium, fault, mistake, failure</em> and many others. We used automatic clustering group those into sets like:

<img decoding="async" alt="Bug Clusters" data-recalc-dims="1" data-src="https://i0.wp.com/computational-linguistics-class.org/assets/img/bug_clusters.jpg?w=980" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/computational-linguistics-class.org/assets/img/bug_clusters.jpg?w=980" alt="Bug Clusters" data-recalc-dims="1">

 </noscript>

These clusters approximate the different word senses of <em>bug</em>. You will explore the main idea underlying our word sense clustering method: which measure the similarity between each pair of paraphrases for a target word and then group together the paraphrases that are most similar to each other. This affinity matrix gives an example of one of the methods for measuring similarity that we tried in <a href="https://www.cis.upenn.edu/~ccb/publications/clustering-paraphrases-by-word-sense.pdf">our paper</a>:

<img decoding="async" alt="Similarity of paraphrses" data-recalc-dims="1" data-src="https://i0.wp.com/computational-linguistics-class.org/assets/img/affinity_matrix.jpg?w=980" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/computational-linguistics-class.org/assets/img/affinity_matrix.jpg?w=980" alt="Similarity of paraphrses" data-recalc-dims="1">

 </noscript>

Here the darkness values give an indication of how similar paraprhases are to each other. For instance <em>sim(insect, pest) &gt; sim(insect, error)</em>.

In this assignment, we will use vector representations in order to measure their similarities of pairs of paraprhases. You will play with different vector space representations of words to create clusters of word senses.

In this image, we have a target word “bug”, and a list of all synonyms (taken from WordNet). The 4 circles are the 4 senses of “bug.” The input to the problem is all the synonyms in a single list, and the task is to separate them correctly. As humans, this is pretty intuitive, but computers aren’t that smart. We will use this task to explore different types of word representations.

You can read more about this task in <a href="https://www.cis.upenn.edu/~ccb/publications/clustering-paraphrases-by-word-sense.pdf">these</a> <a href="https://cs.uwaterloo.ca/~cdimarco/pdf/cs886/Pantel+Lin02.pdf">papers</a>.

<h1 id="clustering-with-word-vectors">Clustering with Word Vectors</h1>

We expect that you have read Jurafsky and Martin, chapters <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">15</a> and <a href="https://web.stanford.edu/~jurafsky/slp3/16.pdf">16</a>. Word vectors, also known as word embeddings, can be thought of simply as points in some high-dimensional space. Remember in geometry class when you learned about the Euclidean plane, and 2-dimensional points in that plane? It’s not hard to understand distance between those points – you can even measure it with a ruler. Then you learned about 3-dimensional points, and how to calculate the distance between these. These 3-dimensional points can be thought of as positions in physical space.

Now, do your best to stop thinking about physical space, and generalize this idea in your mind: you can calculate a distance between 2-dimensional and 3-dimensional points, now imagine a point with 300 dimensions. The dimensions don’t necessarily have meaning in the same way as the X,Y, and Z dimensions in physical space, but we can calculate distances all the same.

This is how we will use word vectors in this assignment: as points in some high-dimensional space, where distances between points are meaningful. The interpretation of distance between word vectors depends entirely on how they were made, but for our purposes, we will consider distance to measure semantic similarity. Word vectors that are close together should have meanings that are similar.

With this framework, we can see how to solve our synonym clustering problem. Imagine in the image below that each point is a (2-dimensional) word vector. Using the distance between points, we can separate them into 3 clusters. This is our task.

<img decoding="async" alt="kmeans" data-src="http://computational-linguistics-class.org/assets/img/kmeans.svg" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="http://computational-linguistics-class.org/assets/img/kmeans.svg" alt="kmeans">

 </noscript> (Image taken from <a href="https://en.wikipedia.org/wiki/K-means_clustering">Wikipedia</a>)

<h2 id="the-data">The Data</h2>

The data to be used for this assignment consists of sets of paraphrases corresponding to one of 56 polysemous target words, e.g.

<table class="table">

 <thead>

  <tr>

   <th scope="col">Target</th>

   <th scope="col">Paraphrase set</th>

  </tr>

 </thead>

 <tbody>

  <tr>

   <td>note.v</td>

   <td>comment mark tell observe state notice say remark mention</td>

  </tr>

  <tr>

   <td>hot.a</td>

   <td>raging spicy blistering red-hot live</td>

  </tr>

 </tbody>

</table>

(Here the <code class="highlighter-rouge">.v</code> following the target <code class="highlighter-rouge">note</code> indicates the part of speech.)

Your objective is to automatically cluster each paraphrase set such that each cluster contains words pertaining to a single <em>sense</em>, or meaning, of the target word. Note that a single word from the paraphrase set might belong to one or more clusters.

For evaluation, we take the set of ground truth senses from <a href="http://wordnet.princeton.edu/">WordNet</a>.

<h3 id="development-data">Development data</h3>

The development data consists of two files – a words file (the input), and a clusters file (to evaluate your output). The words file <code class="highlighter-rouge">dev_input.txt</code> is formatted such that each line contains one target, its paraphrase set, and the number of ground truth clusters <em>k</em>, separated by a <code class="highlighter-rouge">::</code> symbol:

You can use <em>k</em> as input to your clustering algorithm.

The clusters file <code class="highlighter-rouge">dev_output.txt</code> contains the ground truth clusters for each target word’s paraphrase set, split over <em>k</em> lines:

<h3 id="test-data">Test data</h3>

For testing, you will receive only words file <code class="highlighter-rouge">test_input.txt</code> containing the test target words and their paraphrase sets. Your job is to create an output file, formatted in the same way as <code class="highlighter-rouge">dev_output.txt</code>, containing the clusters produced by your system. Neither order of senses, nor order of words in a cluster matter.

<h2 id="evaluation">Evaluation</h2>

There are many possible ways to evaluate clustering solutions. For this homework we will rely on the paired F-score, which you can read more about in <a href="https://www.cs.york.ac.uk/semeval2010_WSI/paper/semevaltask14.pdf">this paper</a>.

The general idea behind paired F-score is to treat clustering prediction like a classification problem; given a target word and its paraphrase set, we call a <em>positive instance</em> any pair of paraphrases that appear together in a ground-truth cluster. Once we predict a clustering solution for the paraphrase set, we similarly generate the set of word pairs such that both words in the pair appear in the same predicted cluster. We can then evaluate our set of predicted pairs against the ground truth pairs using precision, recall, and F-score.

We have provided an evaluation script that you can use when developing your own system. You can run it as follows:

<h2 id="baselines">Baselines</h2>

On the dev data, a random baseline gets about 20%, the word cooccurrence matrix gets about 36%, and the word2vec vectors get about 30%.

<h3 id="1-sparse-representations">1. Sparse Representations</h3>

Your next task is to generate clusters for the target words in <code class="highlighter-rouge">test_input.txt</code> based on a feature-based (not dense) vector space representation. In this type of VSM, each dimension of the vector space corresponds to a specific feature, such as a context word (see, for example, the term-context matrix described in <a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Chapter 15.1.2 of Jurafsky &amp; Martin</a>).

You will calculate cooccurrence vectors on the Reuters RCV1 corpus. Download a <a href="https://www.cis.upenn.edu/~cis530/18sp/data/reuters.rcv1.tokenized.gz">tokenized and cleaned version here</a>. The original is <a href="https://archive.ics.uci.edu/ml/datasets/Reuters+RCV1+RCV2+Multilingual,+Multiview+Text+Categorization+Test+collection">here</a>. Use the provided script, <code class="highlighter-rouge">makecooccurrences.py</code>, to build these vectors. Be sure to set D and W to what you want.

It can take a long time to build cooccurrence vectors, so we have pre-built a set, included in the data.zip, called <code class="highlighter-rouge">coocvec-500mostfreq-window-3.vec.filter</code>. To save on space, these include only the words used in the given files.

You will add K-means clustering to <code class="highlighter-rouge">vectorcluster.py</code>. Here is an example of the K-means code:

<figure class="highlight">

 <pre><code class="language-python" data-lang="python"><span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KMeans</span><span class="n">kmeans</span> <span class="o">=</span> <span class="n">KMeans</span><span class="p">(</span><span class="n">n_clusters</span><span class="o">=</span><span class="n">k</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">)</span><span class="k">print</span><span class="p">(</span><span class="n">kmeans</span><span class="o">.</span><span class="n">labels_</span><span class="p">)</span></code></pre>

</figure>

The baseline system for this section represents words using a term-context matrix <code class="highlighter-rouge">M</code> of size <code class="highlighter-rouge">|V| x D</code>, where <code class="highlighter-rouge">|V|</code> is the size of the vocabulary and D=500. Each feature corresponds to one of the top 500 most-frequent words in the corpus. The value of matrix entry <code class="highlighter-rouge">M[i][j]</code> gives the number of times the context word represented by column <code class="highlighter-rouge">j</code> appeared within W=3 words to the left or right of the word represented by row <code class="highlighter-rouge">i</code> in the corpus. Using this representation, the baseline system clusters each paraphrase set using K-means.

While experimenting, write out clusters for the dev input to <code class="highlighter-rouge">dev_output_features.txt</code> and use the <code class="highlighter-rouge">evaluate.py</code> script to compare against the provided <code class="highlighter-rouge">dev_output.txt</code>.

Implementing the baseline will score you a B, but why not try and see if you can do better? You might try experimenting with different features, for example:

<ul>

 <li>What if you reduce or increase <code class="highlighter-rouge">D</code> in the baseline implementation?</li>

 <li>Does it help to change the window <code class="highlighter-rouge">W</code> used to extract contexts?</li>

 <li>Play around with the feature weighting – instead of raw counts, would it help to use PPMI?</li>

 <li>Try a different clustering algorithm that’s included with the <a href="http://scikit-learn.org/stable/modules/clustering.html">scikit-learn clustering package</a>, or implement your own.</li>

 <li>What if you include additional types of features, like paraphrases in the <a href="http://www.paraphrase.org/">Paraphrase Database</a> or the part-of-speech of context words?</li>

</ul>

The only feature types that are off-limits are WordNet features.

Turn in the predicted clusters that your VSM generates in the file <code class="highlighter-rouge">test_output_features.txt</code>. Also provide a brief description of your method in <code class="highlighter-rouge">writeup.pdf</code>, making sure to describe the vector space model you chose, the clustering algorithm you used, and the results of any preliminary experiments you might have run on the dev set. We have provided a LaTeX file shell, <code class="highlighter-rouge">writeup.tex</code>, which you can use to guide your writeup.

<h3 id="2-dense-representations">2. Dense Representations</h3>

Finally, we’d like to see if dense word embeddings are better for clustering the words in our test set. Run the word clustering task again, but this time use a dense word representation.

For this task, use files:

<ul>

 <li><a href="https://code.google.com/archive/p/word2vec/">Google’s pretrained word2vec vectors</a>, under the heading “Pretrained word and phrase vectors”</li>

 <li>The Google file is very large (~3.4GB), so we have also included in the data.zip a file called <code class="highlighter-rouge">GoogleNews-vectors-negative300.filter</code>, which is filtered to contain only the words in the dev/test splits.</li>

 <li>Modify <code class="highlighter-rouge">vectorcluster.py</code> to load dense vectors.</li>

</ul>

The baseline system for this section uses the provided word vectors to represent words, and K-means for clustering.

As before, achieving the baseline score will get you a B, but you might try to see if you can do better. Here are some ideas:

<ul>

 <li>Try downloading a different dense vector space model from the web, like <a href="https://www.cs.cmu.edu/~jwieting/">Paragram</a> or <a href="https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md">fastText</a>.</li>

 <li>Train your own word vectors, either on the provided corpus or something you find online. You can use the <code class="highlighter-rouge">gensim.models.Word2Vec</code> package for the skip-gram or CBOW models, or <a href="https://nlp.stanford.edu/projects/glove/">GLOVE</a>. Try experimenting with the dimensionality.</li>

 <li><a href="https://www.cs.cmu.edu/~hovy/papers/15HLT-retrofitting-word-vectors.pdf">Retrofitting</a> is a simple way to add additional semantic knowledge to pre-trained vectors. The retrofitting code is available <a href="https://github.com/mfaruqui/retrofitting">here</a>. Experiment with different lexicons, or even try <a href="http://www.aclweb.org/anthology/N16-1018">counter-fitting</a>.</li>

</ul>

As in question 2, turn in the predicted clusters that your dense vector representation generates in the file <code class="highlighter-rouge">test_output_dense.txt</code>. Also provide a brief description of your method in <code class="highlighter-rouge">writeup.pdf</code> that includes the vectors you used, and any experimental results you have from running your model on the dev set.

In addition, do an analysis of different errors made by each system – i.e. look at instances that the word-context matrix representation gets wrong and dense gets right, and vice versa, and see if there are any interesting patterns. There is no right answer for this.

<h3 id="3-the-leaderboard">3. The Leaderboard</h3>

In order to stir up some friendly competition, we would also like you to submit the clustering from your best model to a leaderboard. Copy the output file from your best model to a file called <code class="highlighter-rouge">test_output_leaderboard.txt</code>, and include it with your submission.

<h3 id="extra-credit">Extra Credit</h3>

We made the clustering problem deliberately easier by providing you with <code class="highlighter-rouge">k</code>, the number of clusters, as an input. But in most clustering situations the best <code class="highlighter-rouge">k</code> isn’t obvious. To take this assignment one step further, see if you can come up with a way to automatically choose <code class="highlighter-rouge">k</code>. We have provided an additional test set, <code class="highlighter-rouge">test_nok_input.txt</code>, where the <code class="highlighter-rouge">k</code> field has been zeroed out. See if you can come up with a method that clusters words by sense, and chooses the best <code class="highlighter-rouge">k</code> on its own. (Don’t look at the number of WordNet synsets for this, as that would ruin all the fun.) The baseline system for this portion always chooses <code class="highlighter-rouge">k=5</code>. You can submit your output to this part in a file called <code class="highlighter-rouge">test_nok_output_leaderboard.txt</code>. Be sure to describe your method in <code class="highlighter-rouge">writeup.pdf</code>.

<h2 id="deliverables">Deliverables</h2>

<h2 id="recommended-readings">Recommended readings</h2>

<table>

 <tbody>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/15.pdf">Vector Semantics.</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="https://web.stanford.edu/~jurafsky/slp3/16.pdf">Semantics with Dense Vectors.</a> Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft) .</td>

  </tr>

  <tr>

   <td><a href="https://arxiv.org/pdf/1301.3781.pdf?">Efficient Estimation of Word Representations in Vector Space.</a> Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. ArXiV 2013. <a class="label label-success" href="http://computational-linguistics-class.org/assignment4.html#efficient-estimation-of-word-representations-abstract" data-toggle="modal">Abstract</a></td>

  </tr>

  <tr>

   <td><a href="https://www.aclweb.org/anthology/N13-1090">Linguistic Regularities in Continuous Space Word Representations.</a> Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig. NAACL 2013. <a class="label label-success" href="http://computational-linguistics-class.org/assignment4.html#linguistic-regularities-in-continous-space-word-representations-abstract" data-toggle="modal">Abstract</a> <a class="label label-default" href="http://computational-linguistics-class.org/assignment4.html#linguistic-regularities-in-continous-space-word-representations-bibtex" data-toggle="modal">BibTex</a></td>

  </tr>

  <tr>

   <td><a href="https://www.semanticscholar.org/paper/Discovering-word-senses-from-text-Pantel-Lin/">Discovering Word Senses from Text.</a> Patrick Pangel and Dekang Ling. KDD 2002. <a class="label label-success" href="http://computational-linguistics-class.org/assignment4.html#discovering-word-senses-from-text-abstract" data-toggle="modal">Abstract</a> <a class="label label-default" href="http://computational-linguistics-class.org/assignment4.html#discovering-word-senses-from-text-bibtex" data-toggle="modal">BibTex</a></td>

  </tr>

  <tr>

   <td><a href="https://www.cis.upenn.edu/~ccb/publications.html">Clustering Paraphrases by Word Sense.</a> Anne Cocos and Chris Callison-Burch. NAACL 2016. <a class="label label-success" href="http://computational-linguistics-class.org/assignment4.html#clustering-paraphrases-by-word-sense-abstract" data-toggle="modal">Abstract</a> <a class="label label-default" href="http://computational-linguistics-class.org/assignment4.html#clustering-paraphrases-by-word-sense-bibtex" data-toggle="modal">BibTex</a></td>

  </tr>

 </tbody>

</table>

5/5 - (1 vote)

In order to use the gensim package, you’ll have to be using Python version 3.6 or higher. On my Mac, I did the following:

<ul>

 <li><code class="highlighter-rouge">brew install python3</code></li>

 <li><code class="highlighter-rouge">pip3 install gensim</code></li>

 <li>Then when I ran python, I used the command <code class="highlighter-rouge">python3</code> instead of just <code class="highlighter-rouge">python</code></li>

</ul>

Here are the materials that you should download for this assignment:

<ul>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/question1.txt"><code class="highlighter-rouge">question1.txt</code></a> A template for answering question 1.</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/data.zip"><code class="highlighter-rouge">data.zip</code></a> Contains all the data</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/vectorcluster.py"><code class="highlighter-rouge">vectorcluster.py</code></a> Main code stub</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/evaluate.py"><code class="highlighter-rouge">evaluate.py</code></a> Evaluation script</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/writeup.tex"><code class="highlighter-rouge">writeup.tex</code></a> Report template.</li>

 <li><a href="http://computational-linguistics-class.org/downloads/hw4/makecooccurrences.py"><code class="highlighter-rouge">makecooccurrences.py</code></a> Script to make cooccurrences (optional use)</li>

 <li><a href="https://www.cis.upenn.edu/~cis530/18sp/data/reuters.rcv1.tokenized.gz">Tokenized Reuters RCV1 Corpus</a></li>

 <li><a href="https://code.google.com/archive/p/word2vec/">Google’s pretrained word2vec vectors</a>, under the heading “Pretrained word and phrase vectors”</li>

</ul>

<pre class="highlight"><code>target.pos :: k :: paraphrase1 paraphrase2 paraphrase3 ...</code></pre>

<pre class="highlight"><code>target.pos :: 1 :: paraphrase2 paraphrase6target.pos :: 2 :: paraphrase3 paraphrase4 paraphrase5...target.pos :: k :: paraphrase1 paraphrase9</code></pre>

<pre class="highlight"><code>python evaluate.py &lt;GROUND-TRUTH-FILE&gt; &lt;PREDICTED-CLUSTERS-FILE&gt;</code></pre>

Here are the deliverables that you will need to submit:

<ul>

 <li><code class="highlighter-rouge">question1.txt</code> file with answers to questions from Exploration</li>

 <li>simple VSM clustering output <code class="highlighter-rouge">test_output_features.txt</code></li>

 <li>dense model clustering output <code class="highlighter-rouge">test_output_dense.txt</code></li>

 <li>your favorite clustering output for the leaderboard, <code class="highlighter-rouge">test_output_leaderboard.txt</code> (this will probably be a copy of either <code class="highlighter-rouge">test_output_features.txt</code> or <code class="highlighter-rouge">test_output_dense.txt</code>)</li>

 <li><code class="highlighter-rouge">writeup.pdf</code> (compiled from <code class="highlighter-rouge">writeup.tex</code>)</li>

 <li>your code (.zip). It should be written in Python 3.</li>

 <li>(optional) the output of your model that automatically chooses the number of clusters, <code class="highlighter-rouge">test_nok_output_leaderboard.txt</code> (submit this to the Gradescope assignment ‘Homework 4 EXTRA CREDIT’)</li>