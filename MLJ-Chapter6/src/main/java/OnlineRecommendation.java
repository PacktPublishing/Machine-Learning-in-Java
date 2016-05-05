import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.PlusAnonymousConcurrentUserDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;


public class OnlineRecommendation {
	Recommender recommender;
	int concurrentUsers = 100;
	int noItems = 10;

	public OnlineRecommendation() throws IOException {

		DataModel model = new StringItemIdFileDataModel(new File("data/chap6/BX-Book-Ratings.csv"), ";");
		PlusAnonymousConcurrentUserDataModel plusModel = new PlusAnonymousConcurrentUserDataModel(model, concurrentUsers);
		// recommender = ...;

	}

	public List<RecommendedItem> recommend(long userId, PreferenceArray preferences) throws TasteException {

		if (userExistsInDataModel(userId)) {
			return recommender.recommend(userId, noItems);
		}
		else {
			PlusAnonymousConcurrentUserDataModel plusModel = (PlusAnonymousConcurrentUserDataModel) recommender.getDataModel();

			// Take an available anonymous user form the poll
			Long anonymousUserID = plusModel.takeAvailableUser();

			// Set temporary preferences
			PreferenceArray tempPrefs = preferences;
			tempPrefs.setUserID(0, anonymousUserID);
			// tempPrefs.setItemID(0, itemID);
			plusModel.setTempPrefs(tempPrefs, anonymousUserID);

			List<RecommendedItem> results = recommender.recommend(anonymousUserID, noItems);

			// Release the user back to the poll
			plusModel.releaseUser(anonymousUserID);

			return results;

		}
	}
	private boolean userExistsInDataModel(long userId) {
		// TODO Auto-generated method stub
		return false;
	}
}
