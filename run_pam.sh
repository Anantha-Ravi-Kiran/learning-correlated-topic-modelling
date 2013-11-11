
if [ $# -ne 3 ]; then
    echo "Usage: <folder containing files/corpus> <output folder> <num_topics>"
	exit 0
fi

echo "start demo"
BASE_FOLDER=`pwd`

if [ -d $1 ]; then
	CORPUS=`basename $1`
else
	CORPUS=$1
fi

OUTPUT_FOLDER=$2 
NUM_OF_TOPICS=$3

for corpus in $CORPUS 
do
	#echo "Converting the files in the folder to UCI format"
	
	#if [ -d $1 ]; then
		##Listing the files in the folder
		#cd $BASE_FOLDER/$1
		#LIST_ABS_PATH="ls -d -1 $PWD/*.*"
		#$LIST_ABS_PATH > $OUTPUT_FOLDER/$corpus.names.txt
		
		##Converting the files to UCI format
		#cd $BASE_FOLDER
		#./tools/extract_files.out $OUTPUT_FOLDER/$corpus.names.txt
		#mv $OUTPUT_FOLDER/M_file.mat $OUTPUT_FOLDER/M_file_$corpus.mat 
		#mv $OUTPUT_FOLDER/vocab.txt $OUTPUT_FOLDER/vocab_$corpus.txt 
	#else
		#echo "downloading UCI $corpus corpus"
		#wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.$corpus.txt -O $OUTPUT_FOLDER/vocab_$corpus.txt
		#wget http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.$corpus.txt.gz -O $OUTPUT_FOLDER/docword.$corpus.txt.gz
		#gunzip -f $OUTPUT_FOLDER/docword.$corpus.txt.gz > $OUTPUT_FOLDER/docword.$corpus.txt
		#mv $OUTPUT_FOLDER/docword.$corpus.txt $OUTPUT_FOLDER/M_file_$corpus.mat
	#fi
	
	##Loading python2.x for anchor word recovery
	#module load python-2.7

	##translating the UCI format to scipy format
    #echo "preprocessing, translate from docword.txt to scipy format"
    #python ./anchor-word-recovery/uci_to_scipy.py $OUTPUT_FOLDER/M_file_$corpus.mat $OUTPUT_FOLDER/M_$corpus.full_docs.mat

	##Removing stop words and rare words
    #echo "preprocessing: removing rare words and stopwords"
    #python ./anchor-word-recovery/truncate_vocabulary.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat $OUTPUT_FOLDER/vocab_$corpus.txt 10

	## Splitting the input into training and testing data
	## Produces output for all the modules mallet, scipy for MoM and CTM
	#python ./tools/test_train_split.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $corpus

	## Generating input (train and test) for Mallet Package 
	#./topic-eval/bin/run ExistingAlphabetImporter --input $OUTPUT_FOLDER/$corpus\_train.mallet --output $OUTPUT_FOLDER/$corpus\_train.in.mallet --vocab $OUTPUT_FOLDER/vocab_$corpus.txt.trunc
	#./topic-eval/bin/run ExistingAlphabetImporter --input $OUTPUT_FOLDER/$corpus\_test.mallet --output $OUTPUT_FOLDER/$corpus\_test.in.mallet --vocab $OUTPUT_FOLDER/vocab_$corpus.txt.trunc

	cp ~/md_prior/input/* $OUTPUT_FOLDER/.


	# Method of Moments
    for loss in L2
    do
        for K in $NUM_OF_TOPICS
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python ./anchor-word-recovery/learn_topics.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat ./anchor-word-recovery/settings.example $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $K $loss $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K
		done
    done

	## LDA
	## Training
	#mkdir  $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS -p
	#./mallet-2.0.7/bin/mallet train-topics --num-topics $NUM_OF_TOPICS --num-threads 10 --input $OUTPUT_FOLDER/$corpus\_train.in.mallet --evaluator-filename $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS/lda.evaluator
	## Evaluation
	#./mallet-2.0.7/bin/mallet evaluate-topics --evaluator $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS/lda.evaluator --input $OUTPUT_FOLDER/$corpus\_test.in.mallet --output-prob $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS/likelihood.test
	#./mallet-2.0.7/bin/mallet evaluate-topics --evaluator $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS/lda.evaluator --input $OUTPUT_FOLDER/$corpus\_train.in.mallet --output-prob $OUTPUT_FOLDER/mallet_lda_$NUM_OF_TOPICS/likelihood.train
	
	# Pachinko Allocation Model
	SUPER_TOPICS=(1 2 5 10) 
	for super_topic in "${SUPER_TOPICS[@]}"
	do
			# Training
			mkdir $OUTPUT_FOLDER/mallet_pam_$NUM_OF_TOPICS\_$super_topic -p
			./mallet-2.0.7/bin/mallet train-topics --num-topics $NUM_OF_TOPICS --num-threads 10  --input $OUTPUT_FOLDER/$corpus\_train.in.mallet --output-ll $OUTPUT_FOLDER/mallet_pam_$NUM_OF_TOPICS\_$super_topic/data_ll_pam.output --use-pam --pam-num-supertopics $super_topic --input-beta  $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K.A --input-test  $OUTPUT_FOLDER/$corpus\_test.in.mallet --model 0
			./mallet-2.0.7/bin/mallet train-topics --num-topics $NUM_OF_TOPICS --num-threads 10  --input $OUTPUT_FOLDER/$corpus\_train.in.mallet --output-ll $OUTPUT_FOLDER/mallet_pam_$NUM_OF_TOPICS\_$super_topic/data_ll_mom_pam.output --use-pam --pam-num-supertopics $super_topic --input-beta  $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K.A --input-test  $OUTPUT_FOLDER/$corpus\_test.in.mallet --model 1
			./mallet-2.0.7/bin/mallet train-topics --num-topics $NUM_OF_TOPICS --num-threads 10  --input $OUTPUT_FOLDER/$corpus\_train.in.mallet --output-ll $OUTPUT_FOLDER/mallet_pam_$NUM_OF_TOPICS\_$super_topic/data_ll_mom_prior_pam.output --use-pam --pam-num-supertopics $super_topic --input-beta  $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K.A --input-test  $OUTPUT_FOLDER/$corpus\_test.in.mallet --model 2
	done	
	
done		
