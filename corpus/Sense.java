
import java.util.*;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetRelation;
import it.uniroma1.lcl.babelnet.BabelSense;
import it.uniroma1.lcl.babelnet.WordNetSynsetID;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.babelnet.data.BabelGloss;
import it.uniroma1.lcl.babelnet.data.BabelExample;
import it.uniroma1.lcl.babelnet.data.BabelPointer;
import it.uniroma1.lcl.babelnet.data.BabelSenseSource;
import it.uniroma1.lcl.jlt.util.Language;

import py4j.GatewayServer;


public class Sense
{
	public String getGlossByWnSynsetId(String wnId)
	{
		BabelNet bn = BabelNet.getInstance();
		BabelSynset by = bn.getSynset(new WordNetSynsetID(wnId));
		if (by == null)
		{
			return "";
		}
		List<BabelGloss> glosses = by.getGlosses(Language.EN);
		String definition = "";
		for (BabelGloss gloss : glosses)
		{
			definition += " " + gloss.getGloss();
		}
		return definition.trim();
	}

	public String getExampleByWnSynsetId(String wnId)
	{
		BabelNet bn = BabelNet.getInstance();
		BabelSynset by = bn.getSynset(new WordNetSynsetID(wnId));
		if (by == null)
		{
			return "";
		}
		List<BabelExample> examples = by.getExamples(Language.EN);
		String ex = "";
		for (BabelExample example : examples)
		{
			ex += " " + example.getExample();
		}
		return ex.trim();
	}

	public List<String> getGlossRelatedWordNetSynsetIds(String wnId)
	{
		BabelNet bn = BabelNet.getInstance();
		BabelSynset by = bn.getSynset(new WordNetSynsetID(wnId));
		if (by == null)
		{
			return new ArrayList<>();
		}

		List<String> wnSynsetIds = new ArrayList<>();

		List<BabelSynsetRelation> disEdges = by.getOutgoingEdges(BabelPointer.GLOSS_DISAMBIGUATED);
		for (BabelSynsetRelation edge : disEdges)
		{
			BabelSynsetID bnId = edge.getBabelSynsetIDTarget();
			BabelSynset rby = bnId.toSynset();
			Optional<BabelSense> rbs = rby.getMainSense(Language.EN);
			if (!rbs.isPresent())
			{
				continue;
			}
			else if (rbs.get().getSource() != BabelSenseSource.WN)
			{
				continue;
			}
			wnSynsetIds.add(rbs.get().getSensekey());
		}

		List<BabelSynsetRelation> monoEdges = by.getOutgoingEdges(BabelPointer.GLOSS_MONOSEMOUS);
		for (BabelSynsetRelation edge : monoEdges)
		{
			BabelSynsetID bnId = edge.getBabelSynsetIDTarget();
			BabelSynset rby = bnId.toSynset();
			Optional<BabelSense> rbs = rby.getMainSense(Language.EN);
			if (!rbs.isPresent())
			{
				continue;
			}
			else if (rbs.get().getSource() != BabelSenseSource.WN)
			{
				continue;
			}
			wnSynsetIds.add(rbs.get().getSensekey());
		}

		return wnSynsetIds;
	}

	static public void main(String[] args)
	{
		try
		{
			Sense app = new Sense();
			GatewayServer server = new GatewayServer(app);
			server.start();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
