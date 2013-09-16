#!/bin/bash
echo "start demo"
BASE_FOLDER=`pwd`

if [ -d $1 ]; then
	CORPUS=`basename $1`
else
	CORPUS=$1
fi

OUTPUT_FOLDER=$2 

for corpus in $CORPUS 
do
	echo "Converting the files in the folder to UCI format"
	
	if [ -d $1 ]; then
		#Listing the files in the folder
		cd $BASE_FOLDER/$1
		LIST_ABS_PATH="ls -d -1 $PWD/*.*"
		$LIST_ABS_PATH > $OUTPUT_FOLDER/$corpus.names.txt
		
		#Converting the files to UCI format
		cd $BASE_FOLDER
		./tools/extract_files.out $OUTPUT_FOLDER/$corpus.names.txt
		mv $OUTPUT_FOLDER/M_file.mat $OUTPUT_FOLDER/M_file_$corpus.mat 
		mv $OUTPUT_FOLDER/vocab.txt $OUTPUT_FOLDER/vocab_$corpus.txt 
	else
		echo "downloading UCI $corpus corpus"
		wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt -O $OUTPUT_FOLDER/vocab_$corpus.txt
		wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz -O $OUTPUT_FOLDER/docword.$corpus.txt.gz
		gunzip -f $OUTPUT_FOLDER/docword.$corpus.txt.gz > $OUTPUT_FOLDER/docword.$corpus.txt
		mv $OUTPUT_FOLDER/docword.$corpus.txt $OUTPUT_FOLDER/M_file_$corpus.mat
	fi
	
	#Loading python2.x for anchor word recovery
	module load python-2.7

	#translating the UCI format to scipy format
    echo "preprocessing, translate from docword.txt to scipy format"
    python ./anchor-word-recovery/uci_to_scipy.py $OUTPUT_FOLDER/M_file_$corpus.mat $OUTPUT_FOLDER/M_$corpus.full_docs.mat

	#Removing stop words and rare words
    echo "preprocessing: removing rare words and stopwords"
    python ./anchor-word-recovery/truncate_vocabulary.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat $OUTPUT_FOLDER/vocab_$corpus.txt 50

	#Converting document in UCI format to mallet package
	echo "Converting document in UCI format mallet package"
	python ./tools/scipy_to_mallet_input.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $OUTPUT_FOLDER/doc.in
	./topic-eval/bin/run ExistingAlphabetImporter --input $OUTPUT_FOLDER/doc.in  --output $OUTPUT_FOLDER/doc.mallet --vocab $OUTPUT_FOLDER/vocab_$corpus.txt.trunc

    for loss in L2
    do
        for K in 50
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python ./anchor-word-recovery/learn_topics.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat ./anchor-word-recovery/settings.example $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $K $loss $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K
		done
    done

	#Loading python3.2 for Topic Modelling module
	#module unload python-2.7
	#module load python-3.2
	
	#Running Expectation maximization ** Running only for K = 50 as it takes long time to run **
	python ./topic-modelling/learn_params.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat $OUTPUT_FOLDER/demo_L2_out.$corpus.50.A  $OUTPUT_FOLDER/demo_L2_out.$corpus.50.topwords $OUTPUT_FOLDER/report $corpus
	gzip -f $OUTPUT_FOLDER/topics_$corpus\_50.txt

	#Running Topic Eval module for computing the log-likelihood - doc-limit 100
	#./topic-eval/bin/run topics.LDAHeldOut --instances  $OUTPUT_FOLDER/doc.mallet --num-topics 50 --parameters $OUTPUT_FOLDER/alpha.txt --topics-files $OUTPUT_FOLDER/topics-50.txt.gz --beta 0.01 --doc-limit 100
    perl ./eval_EM.pl $OUTPUT_FOLDER/alpha_$corpus\_50.txt $OUTPUT_FOLDER/pi_$corpus\_50.txt 5 2 $corpus $OUTPUT_FOLDER
done
