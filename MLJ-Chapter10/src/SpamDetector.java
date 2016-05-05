import java.util.ArrayList;

import cc.mallet.types.*;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;

import java.util.regex.*;
import java.io.*;

public class SpamDetector {

	public static void main(String[] args){
		
    	String stopListFilePath = "data/stoplists/en.txt";
    	String dataFolderPath = "data/ex6DataEmails/train";
    	String testFolderPath = "data/ex6DataEmails/test";
    	
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Input2CharSequence("UTF-8"));
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		pipeList.add(new CharSequence2TokenSequence(tokenPattern));
		pipeList.add(new TokenSequenceLowercase());
		pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false));
		pipeList.add(new TokenSequence2FeatureSequence());
		pipeList.add(new FeatureSequence2FeatureVector());
		pipeList.add(new Target2Label());
		SerialPipes pipeline = new SerialPipes(pipeList);
		
		FileIterator folderIterator = new FileIterator(
				new File[] {new File(dataFolderPath)},
		         new TxtFilter(),
		         FileIterator.LAST_DIRECTORY);

		
		InstanceList instances = new InstanceList(pipeline);
		
		instances.addThruPipe(folderIterator);
		
		ClassifierTrainer classifierTrainer = new NaiveBayesTrainer();
		Classifier classifier = classifierTrainer.train(instances);

		InstanceList testInstances = new InstanceList(classifier.getInstancePipe());
		folderIterator = new FileIterator(
				new File[] {new File(testFolderPath)},
		         new TxtFilter(),
		         FileIterator.LAST_DIRECTORY);
        testInstances.addThruPipe(folderIterator);
        
        Trial trial = new Trial(classifier, testInstances);
        
        System.out.println("Accuracy: " + trial.getAccuracy());
        System.out.println("F1 for class 'spam': " + trial.getF1("spam"));

        System.out.println("Precision for class '" +
                           classifier.getLabelAlphabet().lookupLabel(1) + "': " +
                           trial.getPrecision(1));

        System.out.println("Recall for class '" +
                           classifier.getLabelAlphabet().lookupLabel(1) + "': " +
                           trial.getRecall(1));

		
		

	}
}
