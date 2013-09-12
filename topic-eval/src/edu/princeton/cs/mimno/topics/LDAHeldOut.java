package edu.princeton.cs.mimno.topics;

import cc.mallet.util.*;
import cc.mallet.types.*;
import cc.mallet.topics.*;

import java.util.zip.*;
import java.util.Arrays;
import java.util.logging.*;
import java.io.*;

public class LDAHeldOut {
	
	protected static Logger logger = MalletLogger.getLogger(LDAHeldOut.class.getName());
	
	static cc.mallet.util.CommandOption.String instancesFile = new cc.mallet.util.CommandOption.String
		(LDAHeldOut.class, "instances", "FILENAME", true, null,
		 "Filename for the instance list used in collecting word co-document statistics", null);
	
	static cc.mallet.util.CommandOption.SpacedStrings wordCountsFiles = new cc.mallet.util.CommandOption.SpacedStrings
		(LDAHeldOut.class, "topics-files", "FILE [FILE ...]", true, null,
		 "Filename for the saved word-topic values", null);

	static cc.mallet.util.CommandOption.String parametersFile = new cc.mallet.util.CommandOption.String
		(LDAHeldOut.class, "parameters", "FILENAME", true, null,
		 "Filename for Dirichlet hyperparameters", null);
	
	static cc.mallet.util.CommandOption.Integer numTopicsOption = new cc.mallet.util.CommandOption.Integer
		(LDAHeldOut.class, "num-topics", "INTEGER", true, 10,
		 "The number of topics", null);
	
	static cc.mallet.util.CommandOption.Integer docLimit = new cc.mallet.util.CommandOption.Integer
		(LDAHeldOut.class, "doc-limit", "INTEGER", true, 0,
		 "Only evaluate the first N documents", null);
	
	static cc.mallet.util.CommandOption.Boolean showWords = new cc.mallet.util.CommandOption.Boolean
		(LDAHeldOut.class, "show-words", "TRUE|FALSE", true, false,
		 "Display the marginal probability of each word in a sequence", null);
	
	static cc.mallet.util.CommandOption.Double topicWordSmoothingOption = new cc.mallet.util.CommandOption.Double
		(LDAHeldOut.class, "beta", "POS NUMBER", true, 0.01,
		 "Dirichlet smoothing on topic-word distributions.", null);
	
	static cc.mallet.util.CommandOption.Integer foldCountOption = new cc.mallet.util.CommandOption.Integer
		(LDAHeldOut.class, "total-folds", "INTEGER", true, 0,
		 "The number of equal-sized held-out cross validation folds. A value 0 will use all data.", null);
	
	static cc.mallet.util.CommandOption.Integer heldOutFoldOption = new cc.mallet.util.CommandOption.Integer
		(LDAHeldOut.class, "held-out-fold", "INTEGER", true, 0,
		 "The index of the cross validation fold to hold out, starting with 0.", null);
	
	SLREvaluator evaluator;
	int numTopics;

	double[] dirichletParameters;
	double dirichletParametersSum = 0;

	int heldOutFold = 0;
	int numHeldOutFolds = 0;

	public LDAHeldOut (Alphabet alphabet, File wordCountsFile, File parametersFile, int numTopics, double beta) throws Exception {
		this.numTopics = numTopics;
		
		double[][] typeWeights = new double [alphabet.size()][];
		int[][] typeTopics = new int[alphabet.size()][];
		double[] tokensPerTopic = new double[numTopics];

		BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(wordCountsFile))));
		String line;
		int wordID = 0;
		while ((line = in.readLine()) != null) {
			if (! line.equals("")) { 
				String[] topicCountPairs = line.split("\t");
				int numNonZeroTopics = topicCountPairs.length;
				typeWeights[wordID] = new double[numNonZeroTopics];
				typeTopics[wordID] = new int[numNonZeroTopics];
				
				int nonZeroPosition = 0;
				for (int pairID = 0; pairID < numNonZeroTopics; pairID++) {
					String[] fields = topicCountPairs[pairID].split(":");
					int topic = Integer.parseInt(fields[0]);
					double weight = Double.parseDouble(fields[1]);

					if (weight > 0.0) {
						typeWeights[wordID][nonZeroPosition] = weight;
						typeTopics[wordID][nonZeroPosition] = topic;
						tokensPerTopic[topic] += weight;
						nonZeroPosition++;
					}
				}
			}
			wordID++;
		}
		in.close();

		dirichletParameters = new double[numTopics];
		Arrays.fill(dirichletParameters, 0.1);
		dirichletParametersSum = numTopics * 0.1;

		if (parametersFile != null) {
			in = new BufferedReader(new FileReader(parametersFile));
			int topic = 0;
			while ((line = in.readLine()) != null) {
				dirichletParameters[topic] = Double.parseDouble(line);
				topic++;
			}
		}

		// Include dummy values for alpha, alphaSum
		evaluator = new SLREvaluator(numTopics, dirichletParameters, dirichletParametersSum, beta, typeWeights, typeTopics, tokensPerTopic);
		evaluator.printWordProbabilities = showWords.value;
	}

	public void setHeldOutFold(int fold, int totalFolds) {
		this.heldOutFold = fold;
		this.numHeldOutFolds = totalFolds;
	}

	public void evaluate(InstanceList testing, PrintWriter out, int limit) {
		int count = 0;

		for (int docIndex = 0; docIndex < testing.size(); docIndex++) {
			if (numHeldOutFolds == 0 || (17 * docIndex) % numHeldOutFolds == heldOutFold) {
				Instance instance = testing.get(docIndex);
				FeatureSequence tokens = (FeatureSequence) instance.getData();

				if (tokens.size() == 0) {
					continue;
				}
				
				double logProb = evaluator.evaluateLeftToRight(instance, 10);
				out.format("%f\t%d\t%f\n", logProb, tokens.size(), logProb / tokens.size());
				count++;
				
				if (limit > 0 && count > limit) { break; }
			}
		}
	}

	public static void main (String[] args) throws Exception {
		CommandOption.setSummary (LDAHeldOut.class,
								  "Evaluate held-out likelihood for given a file of word-topic counts (possibly floating point).");
		CommandOption.process (LDAHeldOut.class, args);

		InstanceList instances = null;
		try {
			instances = InstanceList.load (new File(instancesFile.value));
		} catch (Exception e) {
			System.err.println("Unable to restore instance list " +
							   instancesFile.value + ": " + e);
			System.exit(1);
		}

		File parameters = null;
		if (parametersFile.value != null) {
			parameters = new File(parametersFile.value);
		}

		for (String wordCountsFile: wordCountsFiles.value) {
			System.out.println(wordCountsFile);

			File outputFile = new File(wordCountsFile + ".heldout");
			if (! outputFile.exists()) {
				
				LDAHeldOut heldOutCalculator = new LDAHeldOut(instances.getDataAlphabet(), new File(wordCountsFile), parameters, numTopicsOption.value, topicWordSmoothingOption.value);
				
				heldOutCalculator.setHeldOutFold(heldOutFoldOption.value, foldCountOption.value);
				PrintWriter out = new PrintWriter(outputFile);
				heldOutCalculator.evaluate(instances, out, docLimit.value);
				out.close();
			}
		}
	}
	
}