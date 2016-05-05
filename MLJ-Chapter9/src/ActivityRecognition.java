import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ActivityRecognition {

	
	public static void main(String[] args) throws Exception{
		
		String databasePath = "data/features.arff";
		
		// Load the data in arff format
		Instances data = new Instances(new BufferedReader(new FileReader(databasePath)));
		
		// Set class the last attribute as class
		data.setClassIndex(data.numAttributes() - 1);

		// Build a basic decision tree model
		String[] options = new String[]{};
		J48 model = new J48();
		model.setOptions(options);
		model.buildClassifier(data);
		
		// Output decision tree
		System.out.println("Decision tree model:\n"+model);
		
		// Output source code implementing the decision tree
		System.out.println("Source code:\n"+model.toSource("ActivityRecognitionEngine"));
		
		// Check accuracy of model using 10-fold cross-validation
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new String[] {});
		System.out.println("Model performance:\n"+eval.toSummaryString());
		
		String[] activities = new String[]{"Walk", "Walk", "Walk", "Run", "Walk", "Run", "Run", "Sit", "Sit", "Sit"};
		DiscreteLowPass dlpFilter = new DiscreteLowPass(3);
		for(String str : activities){
			System.out.println(str +" -> "+ dlpFilter.filter(str));
		}
		
	}
	
}
class DiscreteLowPass{
	
	List<Object> last;
	int window;
	
	
	public DiscreteLowPass(int window){
		this.last = new ArrayList<Object>();
		this.window = window;
	}
	
	public Object filter(Object obj){
		if(last.size() < window){
			last.add(obj);
			return obj;
		}
		
		boolean same = true;
		for(Object o : last){
			if(!o.equals(obj)){
				same = false;
			}
		}
		if(same){
			return obj;
		}
		else{
			Object o = getMostFrequentElement(last);
			last.add(obj);
			last.remove(0);
			return o;
		}
	}
	
	private Object getMostFrequentElement(List<Object> list){
		
		HashMap<String, Integer> objectCounts = new HashMap<String, Integer>();
		Integer frequntCount = 0;
		Object frequentObject = null;
		
		for(Object obj : list){
			String key = obj.toString();
			Integer count = objectCounts.get(key);
			if(count == null){
				count = 0;
			}
			objectCounts.put(key, ++count);
			
			if(count >= frequntCount){
				frequntCount = count;
				frequentObject = obj;
			}
		}
		
		return frequentObject;
	}
	
}


