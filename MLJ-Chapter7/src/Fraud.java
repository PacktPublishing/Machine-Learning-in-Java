

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Fraud {
	
	public static void main(String[] args) throws Exception{
		
		String filePath = "data/claims.csv";
		
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(filePath));
		Instances data = loader.getDataSet();
		
		/* 
		 * Configure the dataset
		 */
		
		// attribute indices
		int CLASS_INDEX = 15;
		int POLICY_INDEX = 17;
		int NO_FRAUD = 0, FRAUD = 1;
		
		int FOLDS = 3;
		
		// covert all to nominal
		NumericToNominal toNominal = new NumericToNominal();
		toNominal.setInputFormat(data);
		data = Filter.useFilter(data, toNominal);
		
		// set class index
		data.setClassIndex(CLASS_INDEX);
		
		// remove policy attribute (it is not relevant)
		Remove remove = new Remove();
		remove.setInputFormat(data);
		remove.setOptions(new String[]{"-R", ""+POLICY_INDEX});
		data = Filter.useFilter(data, remove);
		
		// print the attributes
		System.out.println(data.toSummaryString());
		System.out.println("Class attribute:\n"+data.attributeStats(data.classIndex()));		
		
		/*
		 * Vanilla approach
		 */
		
		// Define classifiers
		ArrayList<Classifier>  models = new ArrayList<Classifier>();
		models.add(new J48());
		models.add(new RandomForest());
		models.add(new NaiveBayes());
		models.add(new AdaBoostM1());
		models.add(new Logistic());

		
		Evaluation eval = new Evaluation(data);
		System.out.println("Vanilla approach\n----------------");
		for(Classifier model : models){
			eval.crossValidateModel(model, data, FOLDS, new Random(1), new String[] {});
			System.out.println(model.getClass().getName() + "\n"+
					"\tRecall:    "+eval.recall(FRAUD) + "\n"+
					"\tPrecision: "+eval.precision(FRAUD) + "\n"+
					"\tF-measure: "+eval.fMeasure(FRAUD));
		}
		
		/* 
		 * Perform manual k-fold cross-validation with dataset rebalancing
		 */
		
		StratifiedRemoveFolds kFold = new StratifiedRemoveFolds();
		kFold.setInputFormat(data);
		
		double measures[][] = new double[models.size()][3];
		
		System.out.println("\nData rebalancing\n----------------");
		for(int k = 1; k <= FOLDS; k++){
		
			// Split data to test and train folds
			kFold.setOptions(new String[]{"-N", ""+FOLDS, "-F", ""+k, "-S", "1"});
			Instances test = Filter.useFilter(data, kFold);
			
			kFold.setOptions(new String[]{"-N", ""+FOLDS, "-F", ""+k, "-S", "1", "-V"});// inverse "-V"
			Instances train = Filter.useFilter(data, kFold);
			
//			System.out.println("Fold "+k+
//					"\n\ttrain: "+train.size()+
//					"\n\ttest: "+ test.size()
//					);
			
			
			// re-balance train dataset 
			Resample resample = new Resample();
			resample.setInputFormat(data);
			resample.setOptions(new String[]{"-Z", "100", "-B", "1"}); //with replacement
			Instances balancedTrain = Filter.useFilter(train, resample); 
			
			for(ListIterator<Classifier> it = models.listIterator(); it.hasNext();){
				Classifier model = it.next();
				model.buildClassifier(balancedTrain);
				eval = new Evaluation(balancedTrain);
				eval.evaluateModel(model, test);
//				System.out.println(
//						"\n\t"+model.getClass().getName() + "\n"+
//						"\tRecall:    "+eval.recall(FRAUD) + "\n"+
//						"\tPrecision: "+eval.precision(FRAUD) + "\n"+
//						"\tF-measure: "+eval.fMeasure(FRAUD));
				// save results for avereage
				measures[it.previousIndex()][0] += eval.recall(FRAUD);
				measures[it.previousIndex()][1] += eval.precision(FRAUD);
				measures[it.previousIndex()][2] += eval.fMeasure(FRAUD);
			}
			
		}
		
		// calculate average
		for(int i = 0; i < models.size(); i++){
			measures[i][0] /= 1.0 * FOLDS;
			measures[i][1] /= 1.0 * FOLDS;
			measures[i][2] /= 1.0 * FOLDS;
		}
		
		// output results and select best model
		Classifier bestModel = null; double bestScore = -1;
		for(ListIterator<Classifier> it = models.listIterator(); it.hasNext();){
			Classifier model = it.next();
			double fMeasure = measures[it.previousIndex()][2];
			System.out.println(
					model.getClass().getName() + "\n"+
					"\tRecall:    "+measures[it.previousIndex()][0] + "\n"+
					"\tPrecision: "+measures[it.previousIndex()][1] + "\n"+
					"\tF-measure: "+fMeasure);
			if(fMeasure > bestScore){
				bestScore = fMeasure;
				bestModel = model;
				
			}
		}
		System.out.println("Best model: "+bestModel.getClass().getName());
		
		// ... build model with all available (resampled) data
		
		
		
		
		
	}

}
