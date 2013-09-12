/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.	For further
   information, see the file `LICENSE' included with this distribution. */

package edu.princeton.cs.mimno.topics;

import java.util.Arrays;
import java.util.ArrayList;

import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

/**
 * An implementation of topic model marginal probability estimators
 *  presented in Wallach et al., "Evaluation Methods for Topic Models", ICML (2009) and 
 *  Buntine, "Estimating Likelihoods for Topic Models"
 *
 * This is the Sequential Left-Right evaluator (SLR).
 * 
 * @author David Mimno
 */

public class SLREvaluator implements Serializable {

	protected int numTopics; // Number of topics to be fit

	protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double alphaSum;
	protected double beta;   // Prior on per-topic multinomial distribution over words
	protected double betaSum;
	
	protected double smoothingOnlyMass = 0.0;
	protected double[] cachedCoefficients;

	double[][] typeWeights;
	int[][] typeTopics;
	double[] tokensPerTopic;

	public boolean printWordProbabilities = false;

	NumberFormat formatter;

	protected Randoms random;
	
	public SLREvaluator (int numTopics,
						 double[] alpha, double alphaSum,
						 double beta,
						 double[][] typeWeights, 
						 int[][] typeTopics, 
						 double[] tokensPerTopic) {
		
		this.numTopics = numTopics;

		this.typeWeights = typeWeights;
		this.typeTopics = typeTopics;
		this.tokensPerTopic = tokensPerTopic;
		
		this.beta = beta;
		this.betaSum = beta * typeTopics.length;
		this.random = new Randoms();

		resetAlpha(alpha, alphaSum);
		
		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);

		System.err.println("Topic Evaluator: " + numTopics + " topics");

	}

	public void resetAlpha(double[] alpha, double alphaSum) {
		this.alpha = alpha;
		this.alphaSum = alphaSum;
		
		cachedCoefficients = new double[ numTopics ];

		// Initialize the smoothing-only sampling bucket
		smoothingOnlyMass = 0;
		
		// Initialize the cached coefficients, using only smoothing.
		//  These values will be selectively replaced in documents with
		//  non-zero counts in particular topics.
		
		for (int topic=0; topic < numTopics; topic++) {
			smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
		}
	}

	public double evaluateLeftToRight (Instance instance, int numSamples, double[] alpha, double alphaSum) {
		this.alpha = alpha;
		this.alphaSum = alphaSum;

		return evaluateLeftToRight(instance, numSamples);
	}

	public double evaluateLeftToRight (Instance instance, int numSamples) {

		double totalLogLikelihood = 0;
			
		FeatureSequence tokenSequence = (FeatureSequence) instance.getData();
		
		double docLogLikelihood = 0;
		
		double[] positionProbabilities = leftToRight(tokenSequence, numSamples);

		for (int position = 0; position < positionProbabilities.length; position++) {
			docLogLikelihood += Math.log(positionProbabilities[position] / numSamples);
			//System.out.print(formatter.format(docLogLikelihood) + ", ");
		}
		//System.out.println();

		if (printWordProbabilities) {
			for (int position = 0; position < positionProbabilities.length; position++) {
				System.out.printf("%s\t%f\n", instance.getDataAlphabet().lookupObject(tokenSequence.getIndexAtPosition(position)), 
								  Math.log(positionProbabilities[position] / numSamples));
				//System.out.print(formatter.format(docLogLikelihood) + ", ");
			}
		}
			
		
		return docLogLikelihood;
	}
	
	public double[] leftToRight (FeatureSequence tokenSequence, int numSamples) {

		int[] oneDocTopics = new int[tokenSequence.getLength()];
		double[] wordProbabilities = new double[tokenSequence.getLength()];

		double[] currentTypeWeights;
		int[] currentTypeTopics;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLength = tokenSequence.getLength();

		// Keep track of the number of tokens we've examined, not
		//  including out-of-vocabulary words
		int tokensSoFar = 0;

		int[] localTopicCounts = new int[numTopics];
		int[] localTopicIndex = new int[numTopics];

		// Build an array that densely lists the topics that
		//  have non-zero counts.
		int denseIndex = 0;

		// Record the total number of non-zero topics
		int nonZeroTopics = denseIndex;

		//		Initialize the topic count/beta sampling bucket
		double topicBetaMass = 0.0;
		double topicTermMass = 0.0;

		double[] topicTermScores = new double[numTopics];
		int[] topicTermIndices;
		int[] topicTermValues;
		int i;
		double score;

		double logLikelihood = 0;

		// All counts are now zero, we are starting completely fresh.

		//	Iterate over the positions (words) in the document 
		for (int limit = 0; limit < docLength; limit++) {
			
			// Record the marginal probability of the token
			//  at the current limit, summed over all topics.

			for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {

				// Iterate up to the current limit
				for (int position = 0; position < limit; position++) {

					type = tokenSequence.getIndexAtPosition(position);
					oldTopic = oneDocTopics[position];

					// Check for out-of-vocabulary words
					if (type >= typeTopics.length ||
						typeTopics[type] == null) {
						continue;
					}

					currentTypeWeights = typeWeights[type];
					currentTypeTopics = typeTopics[type];
				
					//	Remove this token from all counts. 
				
					// Remove this topic's contribution to the 
					//  normalizing constants.
					// Note that we are using clamped estimates of P(w|t),
					//  so we are NOT changing smoothingOnlyMass.
					topicBetaMass -= beta * localTopicCounts[oldTopic] /
						(tokensPerTopic[oldTopic] + betaSum);
				
					// Decrement the local doc/topic counts
				
					localTopicCounts[oldTopic]--;
				
					// Maintain the dense index, if we are deleting
					//  the old topic
					if (localTopicCounts[oldTopic] == 0) {
					
						// First get to the dense location associated with
						//  the old topic.
					
						denseIndex = 0;
					
						// We know it's in there somewhere, so we don't 
						//  need bounds checking.
						while (localTopicIndex[denseIndex] != oldTopic) {
							denseIndex++;
						}
					
						// shift all remaining dense indices to the left.
						while (denseIndex < nonZeroTopics) {
							if (denseIndex < localTopicIndex.length - 1) {
								localTopicIndex[denseIndex] = 
									localTopicIndex[denseIndex + 1];
							}
							denseIndex++;
						}
					
						nonZeroTopics --;
					}

					// Add the old topic's contribution back into the
					//  normalizing constants.
					topicBetaMass += beta * localTopicCounts[oldTopic] /
						(tokensPerTopic[oldTopic] + betaSum);

					// Reset the cached coefficient for this topic
					cachedCoefficients[oldTopic] = 
						(alpha[oldTopic] + localTopicCounts[oldTopic]) /
						(tokensPerTopic[oldTopic] + betaSum);
				

					// Now go over the type/topic counts, calculating the score
					//  for each topic.
				
					int index = 0;
					int currentTopic;
					double currentValue;
				
					boolean alreadyDecremented = false;
				
					topicTermMass = 0.0;
				
					while (index < currentTypeWeights.length && 
						   currentTypeWeights[index] > 0.0) {
						currentTopic = currentTypeTopics[index];
						currentValue = currentTypeWeights[index];
					
						score = 
							cachedCoefficients[currentTopic] * currentValue;
						topicTermMass += score;
						topicTermScores[index] = score;
					
						index++;
					}
			
					double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
					double origSample = sample;
				
					//	Make sure it actually gets set
					newTopic = -1;
				
					if (sample < topicTermMass) {
					
						i = -1;
						while (sample > 0) {
							i++;
							sample -= topicTermScores[i];
						}
					
						newTopic = currentTypeTopics[i];
					}
					else {
						sample -= topicTermMass;
					
						if (sample < topicBetaMass) {
							//betaTopicCount++;
						
							sample /= beta;
						
							for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
								int topic = localTopicIndex[denseIndex];
							
								sample -= localTopicCounts[topic] /
									(tokensPerTopic[topic] + betaSum);
							
								if (sample <= 0.0) {
									newTopic = topic;
									break;
								}
							}
						
						}
						else {
							//smoothingOnlyCount++;
						
							sample -= topicBetaMass;
						
							sample /= beta;
						
							newTopic = 0;
							sample -= alpha[newTopic] /
								(tokensPerTopic[newTopic] + betaSum);
						
							while (sample > 0.0) {
								newTopic++;
								sample -= alpha[newTopic] / 
									(tokensPerTopic[newTopic] + betaSum);
							}
						
						}
					
					}
				
					if (newTopic == -1) {
						System.err.println("sampling error: "+ origSample + " " + sample + " " + smoothingOnlyMass + " " + 
										   topicBetaMass + " " + topicTermMass);
						newTopic = numTopics-1; // TODO is this appropriate
						//throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
					}
					//assert(newTopic != -1);
				
					//			Put that new topic into the counts
					oneDocTopics[position] = newTopic;
				
					topicBetaMass -= beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
					localTopicCounts[newTopic]++;
				
					// If this is a new topic for this document,
					//  add the topic to the dense index.
					if (localTopicCounts[newTopic] == 1) {
					
						// First find the point where we 
						//  should insert the new topic by going to
						//  the end (which is the only reason we're keeping
						//  track of the number of non-zero
						//  topics) and working backwards
					
						denseIndex = nonZeroTopics;
					
						while (denseIndex > 0 &&
							   localTopicIndex[denseIndex - 1] > newTopic) {
						
							localTopicIndex[denseIndex] =
								localTopicIndex[denseIndex - 1];
							denseIndex--;
						}
					
						localTopicIndex[denseIndex] = newTopic;
						nonZeroTopics++;
					}
				
					//	update the coefficients for the non-zero topics
					cachedCoefficients[newTopic] =
						(alpha[newTopic] + localTopicCounts[newTopic]) /
						(tokensPerTopic[newTopic] + betaSum);
				
					topicBetaMass += beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
				}
			
				// We've just resampled all tokens UP TO the current limit.
				//  Now gather the marginal prob of this token, which is
				//   the sum of the various sampling constants.
				
				type = tokenSequence.getIndexAtPosition(limit);				
				
				// Check for out-of-vocabulary words
				if (type >= typeTopics.length ||
					typeTopics[type] == null) {
					continue;
				}
				
				currentTypeWeights = typeWeights[type];
				currentTypeTopics = typeTopics[type];
				
				int index = 0;
				int currentTopic;
				double currentValue;
				
				topicTermMass = 0.0;
				
				while (index < currentTypeWeights.length && 
					   currentTypeWeights[index] > 0.0) {
					currentTopic = currentTypeTopics[index];
					currentValue = currentTypeWeights[index];
					
					score = 
						cachedCoefficients[currentTopic] * currentValue;
					topicTermMass += score;
					topicTermScores[index] = score;
					
					//System.out.println("  " + currentTopic + " = " + currentValue);
					
					index++;
				}
			
				// Note that we've been absorbing (alphaSum + docLength) into
				//  the normalizing constant. The true marginal probability needs
				//  this term, so we stick it back in.
				wordProbabilities[limit] += (smoothingOnlyMass + topicBetaMass + topicTermMass) / (alphaSum + tokensSoFar);
				
				// if this is the last sample for this word, then sample a value for the 
				//  current word before moving back to the beginning of the loop with the
				//  next position.

				if (sampleIndex == numSamples - 1) {
					
					double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
					double origSample = sample;
				
					//System.out.println("normalizer: " + alphaSum + " + " + tokensSoFar);
					tokensSoFar++;
				
					//	Make sure it actually gets set
					newTopic = -1;
				
					if (sample < topicTermMass) {
						i = -1;
						while (sample > 0) {
							i++;
							sample -= topicTermScores[i];
						}
					
						newTopic = currentTypeTopics[i];
					}
					else {
						sample -= topicTermMass;
					
						if (sample < topicBetaMass) {
							sample /= beta;
						
							for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
								int topic = localTopicIndex[denseIndex];
							
								sample -= localTopicCounts[topic] /
									(tokensPerTopic[topic] + betaSum);
							
								if (sample <= 0.0) {
									newTopic = topic;
									break;
								}
							}
						}
						else {
							sample -= topicBetaMass;
							sample /= beta;
						
							newTopic = 0;
							sample -= alpha[newTopic] /
								(tokensPerTopic[newTopic] + betaSum);
						
							while (sample > 0.0) {
								newTopic++;
								sample -= alpha[newTopic] / 
									(tokensPerTopic[newTopic] + betaSum);
							}
						}
					}
				
					if (newTopic == -1) {
						System.err.println("sampling error: "+ origSample + " " + 
										   sample + " " + smoothingOnlyMass + " " + 
										   topicBetaMass + " " + topicTermMass);
						newTopic = numTopics-1; // TODO is this appropriate
					}
				
					// Put that new topic into the counts
					oneDocTopics[limit] = newTopic;
				
					topicBetaMass -= beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
					localTopicCounts[newTopic]++;
				
					// If this is a new topic for this document,
					//  add the topic to the dense index.
					if (localTopicCounts[newTopic] == 1) {
					
						// First find the point where we 
						//  should insert the new topic by going to
						//  the end (which is the only reason we're keeping
						//  track of the number of non-zero
						//  topics) and working backwards
					
						denseIndex = nonZeroTopics;
					
						while (denseIndex > 0 &&
							   localTopicIndex[denseIndex - 1] > newTopic) {
						
							localTopicIndex[denseIndex] =
								localTopicIndex[denseIndex - 1];
							denseIndex--;
						}
					
						localTopicIndex[denseIndex] = newTopic;
						nonZeroTopics++;
					}
				
					//	update the coefficients for the non-zero topics
					cachedCoefficients[newTopic] =
						(alpha[newTopic] + localTopicCounts[newTopic]) /
						(tokensPerTopic[newTopic] + betaSum);
				
					topicBetaMass += beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
					//System.out.println(type + "\t" + newTopic + "\t" + logLikelihood);
				}
			}
		}

		//	Clean up our mess: reset the coefficients to values with only
		//	smoothing. The next doc will update its own non-zero topics...

		for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
			int topic = localTopicIndex[denseIndex];

			cachedCoefficients[topic] =
				alpha[topic] / (tokensPerTopic[topic] + betaSum);
		}

		return wordProbabilities;

	}

	private static final long serialVersionUID = 1;
    private static final int CURRENT_SERIAL_VERSION = 0;
    private static final int NULL_INTEGER = -1;

    private void writeObject (ObjectOutputStream out) throws IOException {
        out.writeInt (CURRENT_SERIAL_VERSION);

        out.writeInt(numTopics);

        out.writeObject(alpha);
        out.writeDouble(alphaSum);
        out.writeDouble(beta);
		out.writeDouble(betaSum);

		out.writeObject(typeWeights);
		out.writeObject(typeTopics);
		out.writeObject(tokensPerTopic);

		out.writeObject(random);

        out.writeDouble(smoothingOnlyMass);
		out.writeObject(cachedCoefficients);
    }

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {

        int version = in.readInt ();

        numTopics = in.readInt();

        alpha = (double[]) in.readObject();
		alphaSum = in.readDouble();
        beta = in.readDouble();
        betaSum = in.readDouble();

        typeWeights = (double[][]) in.readObject();
        typeTopics = (int[][]) in.readObject();
        tokensPerTopic = (double[]) in.readObject();

        random = (Randoms) in.readObject();

        smoothingOnlyMass = in.readDouble();
        cachedCoefficients = (double[]) in.readObject();
    }

    public static SLREvaluator read (File f) throws Exception {

        SLREvaluator estimator = null;

        ObjectInputStream ois = new ObjectInputStream (new FileInputStream(f));
        estimator = (SLREvaluator) ois.readObject();
        ois.close();

        return estimator;
    }


}
