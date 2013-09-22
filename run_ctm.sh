
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
    python ./anchor-word-recovery/truncate_vocabulary.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat $OUTPUT_FOLDER/vocab_$corpus.txt 10

	# Splitting the input into training and testing data
	# Produces output for all the modules mallet, scipy for MoM and CTM
	python ./tools/test_train_split.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $corpus

	# Method of Moments
    for loss in L2
    do
        for K in $NUM_OF_TOPICS
        do
            echo "learning with nonnegative recover method using $loss loss..."
            python ./anchor-word-recovery/learn_topics.py $OUTPUT_FOLDER/M_$corpus.full_docs.mat.trunc.mat ./anchor-word-recovery/settings.example $OUTPUT_FOLDER/vocab_$corpus.txt.trunc $K $loss $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K
		done
    done

	# Correlated Topic Modelling
	# Training
	./ctm/ctm est $OUTPUT_FOLDER/$corpus\_train.ctm $NUM_OF_TOPICS seed $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS ./ctm/settings.txt
	# Held-out likelihood
	./ctm/ctm inf $OUTPUT_FOLDER/$corpus\_test.ctm  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/final  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/heldout_test ./ctm/inf-settings.txt
	# Data likelihood
	./ctm/ctm inf $OUTPUT_FOLDER/$corpus\_train.ctm  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/final  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/heldout_train ./ctm/inf-settings.txt		
	# Computing average Likelihood
	python ./tools/compute_joint_ll.py  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/heldout_test-ctm-lhood.dat > $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/likelihood.test
	python ./tools/compute_joint_ll.py  $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/heldout_train-ctm-lhood.dat > $OUTPUT_FOLDER/ctm_$NUM_OF_TOPICS/likelihood.train
	
	# MoM-CTM
	./ctm/ctm est $OUTPUT_FOLDER/$corpus\_train.ctm $NUM_OF_TOPICS mom_init $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS ./ctm/settings.txt $OUTPUT_FOLDER/demo_$loss\_out.$corpus.$K.A
	## Held-out likelihood
	./ctm/ctm inf $OUTPUT_FOLDER/$corpus\_test.ctm  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/final  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/heldout_test ./ctm/inf-settings.txt
	# Data likelihood
	./ctm/ctm inf $OUTPUT_FOLDER/$corpus\_train.ctm  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/final  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/heldout_train ./ctm/inf-settings.txt		
	# Computing average Likelihood
	python ./tools/compute_joint_ll.py  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/heldout_test-ctm-lhood.dat > $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/likelihood.test
	python ./tools/compute_joint_ll.py  $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/heldout_train-ctm-lhood.dat > $OUTPUT_FOLDER/mom_ctm_$NUM_OF_TOPICS/likelihood.train
	
done		
