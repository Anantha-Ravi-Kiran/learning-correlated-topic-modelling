#!/opt/local/bin/perl
use strict; 
use warnings;


die "Usage: $0 alpha_file pi_file num_components num_iterations corpus_name output_folder num_topics input_file\n" if @ARGV < 6;

my $PARAMS_FILE = $ARGV[0];
my $WEIGHTS_FILE = $ARGV[1];
my $NUM_COMPONENTS= $ARGV[2];
my $NUM_ITERATIONS= $ARGV[3];
my $corpus = $ARGV[4];
my $OUTPUT_FOLDER = $ARGV[5];
my $NUM_TOPICS = $ARGV[6];
my $INSTANCES = $ARGV[7];
my $cleanup = 1;

open(IN, $PARAMS_FILE) or die("Could not open parameter file.");
open(WEIGHTS, $WEIGHTS_FILE) or die("Could not open weights file.");
open (EVAL_OUT, "> $INSTANCES.EM_eval");

my @weights;
my $w;

for (my $iter=0;$iter<$NUM_ITERATIONS;$iter++)
{
    for (my $component=0; $component<$NUM_COMPONENTS; $component++)
    {
        #read in component weight
        $w = <WEIGHTS>;
        chomp($w);
        $weights[$component] = $w;

        #read in parameters
        my $params = <IN>;
        $params =~ s/\t/\n/g;
        chomp($params);

        #format them in a new file
        open (OUT, ">$OUTPUT_FOLDER/alpha_$NUM_TOPICS\_$iter\_$component.txt");
        print OUT $params;
        close(OUT); 

        #send file to java code to evaluate the likelihood
        my $cmd = "./topic-eval/bin/run topics.LDAHeldOut --instances $INSTANCES --num-topics $NUM_TOPICS --parameters $OUTPUT_FOLDER/alpha_$NUM_TOPICS\_$iter\_$component.txt --topics-files $OUTPUT_FOLDER/topics_$corpus\_$NUM_TOPICS.txt.gz --beta 0.01 --doc-limit 100 ";
        `$cmd`;

        `mv $OUTPUT_FOLDER/topics_$corpus\_$NUM_TOPICS.txt.gz.heldout $OUTPUT_FOLDER/topics_$NUM_TOPICS\_$iter\_$component.txt.gz.heldout`;

        #cleanup
        if($cleanup)
        {
            `rm $OUTPUT_FOLDER/alpha_$NUM_TOPICS\_$iter\_$component.txt`;
        }
    }

    #do averaging
    #paste together evaluation files
    my $cmd = "paste $OUTPUT_FOLDER/topics_$NUM_TOPICS\_$iter\_*.txt.gz.heldout | cut -f ";
    $cmd .= " 2,";
    for (my $component=0; $component<$NUM_COMPONENTS; $component++)
    {
        $cmd .= 3*($component)+3;
        $cmd .= ",";
    }
    chop($cmd);
    my $str = `$cmd`;

    #cleanup individual evaluation files
    if($cleanup)
    {
        `rm $OUTPUT_FOLDER/topics_$NUM_TOPICS\_$iter\_*.txt.gz.heldout`;
    }
    
    #each line evaluates a document
    my @lines = split /\n/, $str;
    my $total = 0;
    my $count = 0;
    foreach my $line (@lines){
        my @vals = split /\t/, $line;
        my $local_total = 0;
        #pattern is numWords, perword likelihood 1, perword likelihood 2, ...
        my $num_words = $vals[0];
        for(my $i=1; $i<=($#vals); $i++)
        {
            $local_total += exp($vals[$i]) * $weights[($i-1)] #weights are the pi values
        }
        $total += $num_words*log($local_total);
        $count += 1;
    }
    my $per_doc_likelihood = $total/$count;
    print "$per_doc_likelihood per document\n";
    print EVAL_OUT "iter $iter: $per_doc_likelihood\n";
}
close (IN); 
close (WEIGHTS);
close (EVAL_OUT);

