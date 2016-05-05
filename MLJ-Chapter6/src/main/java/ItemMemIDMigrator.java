import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.AbstractIDMigrator;


public class ItemMemIDMigrator extends AbstractIDMigrator {

	private FastByIDMap<String> longToString;

	public ItemMemIDMigrator() {
		this.longToString = new FastByIDMap<String>(10000);
	}

	public void storeMapping(long longID, String stringID) {
		longToString.put(longID, stringID);
	}

	public void singleInit(String stringID) throws TasteException {
		storeMapping(toLongID(stringID), stringID);
	}

	public String toStringID(long longID) {
		return longToString.get(longID);
	}

}