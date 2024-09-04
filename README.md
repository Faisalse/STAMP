<!DOCTYPE html>
<html>
<head>

</head>
<body>

<h2>Installation guide</h2>  

<h5>Using Anaconda</h5>
  <ul>
    <li>Download Anaconda from <a href="https://www.anaconda.com/">https://www.anaconda.com/</a> and install it</li>
    <li>Clone the GitHub repository by using this link: <code>https://github.com/Faisalse/STAMP.git</code></li>
    <li>Open the Anaconda command prompt</li>
    <li>Move into the <b>STAMP</b> directory</li>
    <li>Run this command to create virtual environment: <code>conda create --name STAMP_env python=3.6</code></li>
    <li>Run this command to activate the virtual environment: <code>conda activate STAMP_env</code></li>
    <li>Run this command to install the required libraries for CPU: <code>pip install -r requirements_stamp.txt</code></li>
  </ul>
</p>

<h5>STAMP and baseline models</h5>
<ul>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Yoochoose</a> dataset, unzip it and put the “yoochoose-clicks.dat” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the STAMP and baseline models on the shorter version of the Yoochoose dataset: <code>python run_experiments_STAMP_baseline_models.py -m stamp_rsc -d rsc15_64 -n</code> and run the following command to create the experiments for the larger version of the Yoochoose dataset <code>python run_experiments_STAMP_baseline_models.py -m stamp_rsc -d rsc15_4 -n</code>  </li>
<li>Download <a href="https://www.dropbox.com/sh/n281js5mgsvao6s/AADQbYxSFVPCun5DfwtsSxeda?dl=0" target="_blank">Diginetica</a> dataset, unzip it and put the “train-item-views.csv” file into the “data” directory/folder. </li>
<li>Run this command to reproduce the experiments for the STAMP and baseline models on the Diginetica dataset: <code>python run_experiments_STAMP_baseline_models.py -m stamp_cikm -d digi -n</code></li> 
</ul>

</body>
</html>  

