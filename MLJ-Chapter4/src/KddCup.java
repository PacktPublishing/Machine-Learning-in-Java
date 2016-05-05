/**
 * Chapter 4: the KDD Cup 2009 challenge 
 * http://kdd.org/kdd-cup/view/kdd-cup-2009
 * 
 * @author Bostjan Kaluza, http://bostjankaluza.net
 */

import java.io.File;
import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.EnsembleLibrary;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.classifiers.meta.EnsembleSelection;
import weka.filters.unsupervised.attribute.Discretize;

public class KddCup {

	// static final Enum;
	static final int 	PREDICT_CHURN = 0, 
						PREDICT_APPETENCY = 1,
						PREDICY_UPSELL = 3;


	public static void main(String args[]) throws Exception {
		
		/*
		 * Build Naive Bayes baseline
		 */


		Classifier baselineNB = new NaiveBayes();

		double resNB[] = evaluate(baselineNB);
		System.out.println("Naive Bayes\n" + 
							"\tchurn:     " + resNB[0] + "\n" + 
							"\tappetency: " + resNB[1] + "\n" + 
							"\tup-sell:   " + resNB[2] + "\n" + 
							"\toverall:   " + resNB[3] + "\n");


		
		/*
		 * Ensemble method
		 */
		
		EnsembleLibrary ensembleLib = new EnsembleLibrary();
		
		// Decision trees
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2");
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A");
		
		// naive Bayes
		ensembleLib.addModel("weka.classifiers.bayes.NaiveBayes");
		
		// k-nn
		ensembleLib.addModel("weka.classifiers.lazy.IBk");

		// AdaBoost
		ensembleLib.addModel("weka.classifiers.meta.AdaBoostM1");
		
		// LogitBoost
		ensembleLib.addModel("weka.classifiers.meta.LogitBoost");

		// SVM
		ensembleLib.addModel("weka.classifiers.functions.SMO");

		// Logistic regression
		ensembleLib.addModel("weka.classifiers.functions.Logistic");
		
		// Simple logistic regression
		ensembleLib.addModel("weka.classifiers.functions.SimpleLogistic");

		
		EnsembleLibrary.saveLibrary(new File("data/ensembleLib.model.xml"), ensembleLib, null);
		System.out.println(ensembleLib.getModels());
		
		EnsembleSelection ensambleSel = new EnsembleSelection();
		ensambleSel.setOptions(new String[]{
				 "-L", "data/ensembleLib.model.xml", // </path/to/modelLibrary> - Specifies the Model Library File, continuing the list of all models.
				 "-W", "data/esTmp", // </path/to/working/directory> - Specifies the Working Directory, where all models will be stored.
				 "-B", "10", // <numModelBags> - Set the number of bags, i.e., number of iterations to run the ensemble selection algorithm.
				 "-E", "1.0", // <modelRatio> - Set the ratio of library models that will be randomly chosen  to populate each bag of models.
				 "-V", "0.25", // <validationRatio> - Set the ratio of the training data set that will be reserved for validation.
				 "-H", "100", // <hillClimbIterations> - Set the number of hillclimbing iterations to be performed on each model bag.
				 "-I", "1.0", // <sortInitialization> - Set the the ratio of the ensemble library that the sort initialization algorithm will be able to choose from while initializing the ensemble for each model bag
				 "-X", "2", // <numFolds> - Sets the number of cross-validation folds.
				 "-P", "roc", // <hillclimbMettric> - Specify the metric that will be used for model selection during the hillclimbing algorithm.
				 "-A", "forward", // <algorithm> - Specifies the algorithm to be used for ensemble selection. 
				 "-R", "true", // - Flag whether or not models can be selected more than once for an ensemble.
				 "-G", "true", // - Whether sort initialization greedily stops adding models when performance degrades.
				 "-O", "true", // - Flag for verbose output. Prints out performance of all selected models.
				 "-S", "1", // <num> - Random number seed.
				 "-D", "true" // - If set, classifier is run in debug mode and may output additional info to the console
		});
		
		double resES[] = evaluate(ensambleSel);
		System.out.println("Ensemble\n" + "\tchurn:     " + resES[0] + "\n"
				+ "\tappetency: " + resES[1] + "\n" + "\tup-sell:   "
				+ resES[2] + "\n" + "\toverall:   " + resES[3] + "\n");
		

	}

	public static double[] evaluate(Classifier model) throws Exception {

		double results[] = new double[4];

		String[] labelFiles = new String[] { "churn", "appetency", "upselling" };

		double overallScore = 0.0;
		for (int i = 0; i < labelFiles.length; i++) {

			// Load data
			Instances train_data = loadData("data/orange_small_train.data",
											"data/orange_small_train_" + labelFiles[i]+ ".labels.txt");
			train_data = preProcessData(train_data);

			// cross-validate the data
			Evaluation eval = new Evaluation(train_data);
			eval.crossValidateModel(model, train_data, 5, new Random(1), new Object[] {});

			// Save results
			results[i] = eval.areaUnderROC(train_data.classAttribute()
					.indexOfValue("1"));
			overallScore += results[i];
			System.out.println(labelFiles[i] + "\t-->\t" +results[i]);
		}
		// Get average results over all three problems
		results[3] = overallScore / 3;
		return results;
	}

	public static Instances preProcessData(Instances data) throws Exception{
		
		/* 
		 * Remove useless attributes
		 */
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" });	// threshold
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		
		/* 
		 * Remove useless attributes
		 */
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);
		

		/* 
		 * Remove useless attributes
		 */
		Discretize discretizeNumeric = new Discretize();
		discretizeNumeric.setOptions(new String[] {
				"-O",
				"-M",  "-1.0", 
				"-B",  "4",  // no of bins
				"-R",  "first-last"}); //range of attributes
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		/* 
		 * Select only informative attributes
		 */
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" });	// information gain threshold
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		
		// apply attribute selection
		attSelect.SelectAttributes(data);
		
		// remove the attributes not selected in the last run
		data = attSelect.reduceDimensionality(data);
		
		

		return data;
	}

	
	public static Instances loadData(String pathData, String pathLabeles)
			throws Exception {

		/*
		 * Load data
		 */
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNominalAttributes("191-last");
		loader.setSource(new File(pathData));
		Instances data = loader.getDataSet();

		// remove String attribute types
		RemoveType removeString = new RemoveType();
		removeString.setOptions(new String[] { "-T", "string" });
		removeString.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, removeString);
		// data.deleteStringAttributes();

		/*
		 * Load labels
		 */
		loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNoHeaderRowPresent(true);
		loader.setNominalAttributes("first-last");
		loader.setSource(new File(pathLabeles));
		Instances labeles = loader.getDataSet();
		// System.out.println(labeles.toSummaryString());

		// Append label as class value
		Instances labeledData = Instances.mergeInstances(filteredData, labeles);

		// set it as a class value
		labeledData.setClassIndex(labeledData.numAttributes() - 1);
		
//		System.out.println(labeledData.toSummaryString());
		
		return labeledData;
	}

}
