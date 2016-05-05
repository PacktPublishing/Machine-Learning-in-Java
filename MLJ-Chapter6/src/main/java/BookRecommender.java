
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;

/**
 * Hello world!
 *
 */
public class BookRecommender implements RecommenderBuilder  {

	static HashMap<String, String> books;

	public static void main(String[] args) throws Exception {

		books = loadBooks("data/BX-Books.csv");

		itemBased();
//		 userBased();
		evaluateRecommender();

	}

	public static HashMap<String, String> loadBooks(String filename) throws Exception {
		HashMap<String, String> map = new HashMap<String, String>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		String line = "";
		while ((line = in.readLine()) != null) {
			String parts[] = line.replace("\"", "").split(";");
			map.put(parts[0], parts[1]);
		}
		in.close();
		// System.out.println(map.toString());
		System.out.println("Total items: " + map.size());
		return map;
	}

	public static ItemBasedRecommender itemBased() throws Exception {

		// Load the data
		StringItemIdFileDataModel dataModel = loadFromFile("data/BX-Book-Ratings.csv", ";");
		// Collection<GenericItemSimilarity.ItemItemSimilarity> correlations =
		// null;
		// ItemItemSimilarity iis = new ItemItemSimilarity(0, 0, 0);
		// ItemSimilarity itemSimilarity = new
		// GenericItemSimilarity(correlations);
		ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(dataModel);

		ItemBasedRecommender recommender = new GenericItemBasedRecommender(
				dataModel, itemSimilarity);

		IDRescorer rescorer = new MyRescorer();

		// List recommendations = recommender.recommend(2, 3, rescorer);
		String itemISBN = "042513976X";
		long itemID = dataModel.readItemIDFromString(itemISBN);
		int noItems = 10;

		System.out.println("Recommendations for item: " + books.get(itemISBN));

		System.out.println("\nMost similar items:");
		List<RecommendedItem> recommendations = recommender.mostSimilarItems(
				itemID, noItems);
		for (RecommendedItem item : recommendations) {
			itemISBN = dataModel.getItemIDAsString(item.getItemID());
			System.out.println("Item: " + books.get(itemISBN) + " | Item id: "
					+ itemISBN + " | Value: " + item.getValue());
		}
		
		return recommender;
	}

	public static void userBased() throws Exception {
		
		StringItemIdFileDataModel model = loadFromFile("data/BX-Book-Ratings.csv",";");
		
		UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
		UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

		IDRescorer rescorer = new MyRescorer();

		// List recommendations = recommender.recommend(2, 3, rescorer);
		long userID = 276704;// 276704;//212124;//277157;
		int noItems = 10;

		System.out.println("Rated items:");
		for (Preference preference : model.getPreferencesFromUser(userID)) {
			String itemISBN = model.getItemIDAsString(preference.getItemID());
			System.out.println("Item: " + books.get(itemISBN) + " | Item id: "
					+ itemISBN + " | Value: " + preference.getValue());
		}

		System.out.println("\nRecommended items:");
		List<RecommendedItem> recommendations = recommender.recommend(userID,
				noItems);
		for (RecommendedItem item : recommendations) {
			String itemISBN = model.getItemIDAsString(item.getItemID());
			System.out.println("Item: " + books.get(itemISBN) + " | Item id: "
					+ itemISBN + " | Value: " + item.getValue());
		}
		

	}

	public static StringItemIdFileDataModel loadFromFile(String filePath, String seperator) throws Exception{
		StringItemIdFileDataModel dataModel = new StringItemIdFileDataModel(
				new File(filePath), seperator);
		return dataModel;
	}
	
	public static DataModel loadFromFile(String filePath) throws IOException{
		// File-based DataModel - FileDataModel
		DataModel dataModel = new FileDataModel(new File("preferences.csv"));
		return dataModel;

	}
	
	public static DataModel loadFromDB() throws Exception {
		
		// Database-based DataModel - MySQLJDBCDataModel
		/*
		 * A JDBCDataModel backed by a PostgreSQL database and accessed via
		 * JDBC. It may work with other JDBC databases. By default, this class
		 * assumes that there is a DataSource available under the JNDI name
		 * "jdbc/taste", which gives access to a database with a
		 * "taste_preferences" table with the following schema: CREATE TABLE
		 * taste_preferences ( user_id BIGINT NOT NULL, item_id BIGINT NOT NULL,
		 * preference REAL NOT NULL, PRIMARY KEY (user_id, item_id) ) CREATE
		 * INDEX taste_preferences_user_id_index ON taste_preferences (user_id);
		 * CREATE INDEX taste_preferences_item_id_index ON taste_preferences
		 * (item_id);
		 */
		MysqlDataSource dbsource = new MysqlDataSource();
		dbsource.setUser("user");
		dbsource.setPassword("pass");
		dbsource.setServerName("localhost");
		dbsource.setDatabaseName("my_db");

		DataModel dataModelDB = new MySQLJDBCDataModel(dbsource,
				"taste_preferences", "user_id", "item_id", "preference",
				"timestamp");
		
		return dataModelDB;
	}

	public DataModel loadInMemory() {
		// In-memory DataModel - GenericDataModels
		FastByIDMap<PreferenceArray> preferences = new FastByIDMap<PreferenceArray>();

		PreferenceArray prefsForUser1 = new GenericUserPreferenceArray(10);
		prefsForUser1.setUserID(0, 1L);
		prefsForUser1.setItemID(0, 101L);
		prefsForUser1.setValue(0, 3.0f);
		prefsForUser1.setItemID(1, 102L);
		prefsForUser1.setValue(1, 4.5F);
		preferences.put(1L, prefsForUser1); // use userID as the key
		
		//TODO: add others users

		
		// Return preferences as new data model
		DataModel dataModel = new GenericDataModel(preferences);
		
		return dataModel;

	}
	
	public static void evaluateRecommender() throws Exception{
		StringItemIdFileDataModel dataModel = loadFromFile("data/BX-Book-Ratings.csv",";");
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new BookRecommender();
		double result = evaluator.evaluate(builder, null, dataModel, 0.9, 1.0);
		System.out.println(result);
	}

	public Recommender buildRecommender(DataModel arg0) {
		try {
			return BookRecommender.itemBased();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

}
