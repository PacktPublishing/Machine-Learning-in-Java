/**
 * Chapter 3: Classification Task
 * 
 * This example shows how to load the data, select features, implement a basic classifier in Weka, 
 * and evaluate classifier performance. This task uses the ZOO dataset. 
 * 
 * @author Bostjan Kaluza, http://bostjankaluza.net
 */

import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;

public class ClassificationTask {

	public static void main(String[] args) throws Exception {

		/*
		 * Load the data
		 */
		DataSource source = new DataSource("data/zoo.arff");
		Instances data = source.getDataSet();
		System.out.println(data.numInstances() + " instances loaded.");
		// System.out.println(data.toString());

		// remove animal attribute
		String[] opts = new String[] { "-R", "1" };
		Remove remove = new Remove();
		remove.setOptions(opts);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);

		/*
		 * Feature selection
		 */
		AttributeSelection attSelect = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(data);
		int[] indices = attSelect.selectedAttributes();
		System.out.println("Selected attributes: "+Utils.arrayToString(indices));

		/*
		 * Build a decision tree
		 */
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);
		System.out.println(tree);

		/*
		 * Classify new instance.
		 */
		double[] vals = new double[data.numAttributes()];
		vals[0] = 1.0; // hair {false, true}
		vals[1] = 0.0; // feathers {false, true}
		vals[2] = 0.0; // eggs {false, true}
		vals[3] = 1.0; // milk {false, true}
		vals[4] = 0.0; // airborne {false, true}
		vals[5] = 0.0; // aquatic {false, true}
		vals[6] = 0.0; // predator {false, true}
		vals[7] = 1.0; // toothed {false, true}
		vals[8] = 1.0; // backbone {false, true}
		vals[9] = 1.0; // breathes {false, true}
		vals[10] = 1.0; // venomous {false, true}
		vals[11] = 0.0; // fins {false, true}
		vals[12] = 4.0; // legs INTEGER [0,9]
		vals[13] = 1.0; // tail {false, true}
		vals[14] = 1.0; // domestic {false, true}
		vals[15] = 0.0; // catsize {false, true}
		Instance myUnicorn = new DenseInstance(1.0, vals);
		//Assosiate your instance with Instance object in this case dataRaw
		myUnicorn.setDataset(data); 

		double label = tree.classifyInstance(myUnicorn);
		System.out.println(data.classAttribute().value((int) label));

		/*
		 * Visualize decision tree
		 */
		TreeVisualizer tv = new TreeVisualizer(null, tree.graph(),
				new PlaceNode2());
		JFrame frame = new javax.swing.JFrame("Tree Visualizer");
		frame.setSize(800, 500);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(tv);
		frame.setVisible(true);
		tv.fitToScreen();

		/*
		 * Evaluation
		 */

		Classifier cl = new J48();
		Evaluation eval_roc = new Evaluation(data);
		eval_roc.crossValidateModel(cl, data, 10, new Random(1), new Object[] {});
		System.out.println(eval_roc.toSummaryString());
		// Confusion matrix
		double[][] confusionMatrix = eval_roc.confusionMatrix();
		System.out.println(eval_roc.toMatrixString());

		/*
		 * Bonus: Plot ROC curve
		 */

		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval_roc.predictions(), classIndex);
		// plot curve
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		vmc.setROCString("(Area under ROC = " + tc.getROCArea(result) + ")");
		vmc.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		// specify which points are connected
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);

		// add plot
		vmc.addPlot(tempd);
		// display curve
		JFrame frameRoc = new javax.swing.JFrame("ROC Curve");
		frameRoc.setSize(800, 500);
		frameRoc.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frameRoc.getContentPane().add(vmc);
		frameRoc.setVisible(true);

	}
}
