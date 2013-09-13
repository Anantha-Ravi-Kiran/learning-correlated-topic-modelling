#!/bin/bash
echo "start demo"
BASE_FOLDER=`pwd`
for corpus in `basename $1` 
do
	echo "Converting the files in the folder to UCI format"
	
	#Listing the files in the folder
	cd $BASE_FOLDER/$1
	LIST_ABS_PATH="ls -d -1 $PWD/*.*"
	$LIST_ABS_PATH > $BASE_FOLDER/output/$corpus.names.txt
	
	#Converting the files to UCI format
	cd $BASE_FOLDER
	./tools/extract_files.out ./output/$corpus.names.txt 
	
	#Loading python2.x for anchor word recovery
	module load python-2.7

	#translating the UCI format to scipy format
    echo "preprocessing, translate from docword.txt to scipy format"
    python ./anchor-word-recovery/uci_to_scipy.py ./output/M_file.mat ./output/M_$corpus.full_docs.mat

	#Removing stop words and rare words
    echo "preprocessing: removing rare words and stopwords"
    python ./anchor-word-recovery/truncate_vocabulary.py ./output/M_$corpus.full_docs.mat ./output/vocab.txt 15

	#Converting document in UCI format to mallet package
	echo "Converting document in UCI format mallet package"
	python ./tools/scipy_to_mallet_input.py ./output/M_$corpus.full_docs.mat.trunc.mat ./output/vocab.txt.trunc ./output/doc.in
	./topic-eval/bin/run ExistingAlphabetImporter --input ./output/doc.in  --output ./output/doc.mallet --vocab ./output/vocab.txt.trunc

    for loss in L2
    do
        for K in 20 50 100
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python ./anchor-word-recovery/learn_topics.py ./output/M_$corpus.full_docs.mat.trunc.mat ./anchor-word-recovery/settings.example ./output/vocab.txt.trunc $K $loss ./output/demo_$loss\_out.$corpus.$K
		done
    done

	#Loading python3.2 for Topic Modelling module
	#module unload python-2.7
	#module load python-3.2
	
	#Running Expectation maximization ** Running only for K = 50 as it takes long time to run **
	python ./topic-modelling/learn_params.py ./output/M_$corpus.full_docs.mat.trunc.mat ./output/demo_L2_out.$corpus.50.A  ./output/demo_L2_out.$corpus.50.topwords ./output/report
	gzip ./output/topics-50.txt

	#Running Topic Eval module for computing the log-likelihood - doc-limit 100
	#./topic-eval/bin/run topics.LDAHeldOut --instances  ./output/doc.mallet --num-topics 50 --parameters ./output/alpha.txt --topics-files ./output/topics-50.txt.gz --beta 0.01 --doc-limit 100
    ./eval_EM.pl ./output/alpha_50.txt ./output/pi_50.txt 5 2
done
