
import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;
import cc.mallet.util.Randoms;

import java.util.*;
import java.util.regex.*;
import java.io.*;

public class TopicModeling {

    public static void main(String[] args) throws Exception {

    	String dataFolderPath = "data/bbc";
    	String stopListFilePath = "data/stoplists/en.txt";

        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
        pipeList.add(new Input2CharSequence("UTF-8"));
        Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
        pipeList.add(new CharSequence2TokenSequence(tokenPattern));
        pipeList.add(new TokenSequenceLowercase());
        pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false));
        pipeList.add(new TokenSequence2FeatureSequence());
        pipeList.add(new Target2Label());
        SerialPipes pipeline = new SerialPipes(pipeList);

        FileIterator folderIterator = new FileIterator(
        			new File[] {new File(dataFolderPath)},
                    new TxtFilter(),
                    FileIterator.LAST_DIRECTORY);
        
        // Construct a new instance list, passing it the pipe
        //  we want to use to process instances.
        InstanceList instances = new InstanceList(pipeline);

        // Now process each instance provided by the iterator.
        instances.addThruPipe(folderIterator);

        // Create a model with 100 topics, alpha_t = 0.01, beta_w = 0.01
        //  Note that the first parameter is passed as the sum over topics, while
        //  the second is the parameter for a single dimension of the Dirichlet prior.
        int numTopics = 5;
        ParallelTopicModel model = new ParallelTopicModel(numTopics, 0.01, 0.01);

        model.addInstances(instances);

        // Use two parallel samplers, which each look at one half the corpus and combine
        //  statistics after every iteration.
        model.setNumThreads(4);

        // Run the model for 50 iterations and stop (this is for testing only, 
        //  for real applications, use 1000 to 2000 iterations)
        model.setNumIterations(50);
        model.estimate();
        
        
        
        
        /*
         * Saving model
         */
        
		String modelPath = "myTopicModel";
		
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream (new File(modelPath+".model")));
		oos.writeObject(model);
		oos.close();     
		oos = new ObjectOutputStream(new FileOutputStream (new File(modelPath+".pipeline")));
		oos.writeObject(pipeline);
		oos.close();     
		
		System.out.println("Model saved.");
		
		/*
		 * Loading the model
		 */
//		ParallelTopicModel model;
//		SerialPipes pipeline;
		ObjectInputStream ois = new ObjectInputStream (new FileInputStream (new File(modelPath+".model")));
		model = (ParallelTopicModel) ois.readObject();
		ois.close();   
		ois = new ObjectInputStream (new FileInputStream (new File(modelPath+".pipeline")));
		pipeline = (SerialPipes) ois.readObject();
		ois.close();   
		
		System.out.println("Model loaded.");
        

        // Show the words and topics in the first instance

        // The data alphabet maps word IDs to strings
        Alphabet dataAlphabet = instances.getDataAlphabet();
        
        FeatureSequence tokens = (FeatureSequence) model.getData().get(0).instance.getData();
        LabelSequence topics = model.getData().get(0).topicSequence;
        
        Formatter out = new Formatter(new StringBuilder(), Locale.US);
        for (int position = 0; position < tokens.getLength(); position++) {
            out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)), topics.getIndexAtPosition(position));
        }
        System.out.println(out);
        
        // Estimate the topic distribution of the first instance, 
        //  given the current Gibbs state.
        double[] topicDistribution = model.getTopicProbabilities(0);

        // Get an array of sorted sets of word ID/count pairs
        ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
        
        // Show top 5 words in topics with proportions for the first document
        for (int topic = 0; topic < numTopics; topic++) {
            Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
            
            out = new Formatter(new StringBuilder(), Locale.US);
            out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
            int rank = 0;
            while (iterator.hasNext() && rank < 5) {
                IDSorter idCountPair = iterator.next();
                out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
                rank++;
            }
            System.out.println(out);
        }
        
        
        
        /*
         * Testing
         */
        
        System.out.println("Evaluation");

		// Split dataset
		InstanceList[] instanceSplit= instances.split(new Randoms(), new double[] {0.9, 0.1, 0.0});
		
		// Use the first 90% for training
		model.addInstances(instanceSplit[0]);
		model.setNumThreads(4);
		model.setNumIterations(50);
		model.estimate();
		
		// Get estimator
		MarginalProbEstimator estimator = model.getProbEstimator();
		double loglike = estimator.evaluateLeftToRight(instanceSplit[1], 10, false, null);//System.out);
		System.out.println("Total log likelihood: "+loglike);



   

}

}
/** This class illustrates how to build a simple file filter */
class TxtFilter implements FileFilter {

    /** Test whether the string representation of the file 
     *   ends with the correct extension. Note that {@ref FileIterator}
     *   will only call this filter if the file is not a directory,
     *   so we do not need to test that it is a file.
     */
    public boolean accept(File file) {
        return file.toString().endsWith(".txt");
    }
}