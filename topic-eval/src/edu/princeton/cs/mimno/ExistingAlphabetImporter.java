package edu.princeton.cs.mimno;

import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.util.*;

import java.util.*;
import java.util.regex.Pattern;
import java.io.*;

/**
 *  This class reads through a single file, breaking each line
 *   into data and (optional) name and label fields.
 */

public class ExistingAlphabetImporter {

	static CommandOption.File inputFile =   new CommandOption.File
		(ExistingAlphabetImporter.class, "input", "FILE", true, null,
		 "The file containing data, one instance per line", null);

	static CommandOption.File outputFile = new CommandOption.File
		(ExistingAlphabetImporter.class, "output", "FILE", true, new File("mallet.data"),
		 "Write the instance list to this file", null);

    static CommandOption.File stoplistFile = new CommandOption.File
		(ExistingAlphabetImporter.class, "stoplist", "FILE", true, null,
		 "Read newline-separated words from this file,\n   and remove them from text. This option overrides\n   the default English stoplist triggered by --remove-stopwords.", null);

    static CommandOption.File alphabetFile = new CommandOption.File
		(ExistingAlphabetImporter.class, "vocab", "FILE", true, null,
		 "Read newline-separated words from this file into the vocabulary.", null);

	static CommandOption.String tokenRegex = new CommandOption.String
		(ExistingAlphabetImporter.class, "token-regex", "REGEX", true, "[\\p{L}_]+",
		  "Regular expression used for tokenization.\n" +
		 "   Example: \"[\\p{L}\\p{N}_]+|[\\p{P}]+\" (unicode letters, numbers and underscore OR all punctuation) ", null);

	static CommandOption.String lineRegex = new CommandOption.String
		(ExistingAlphabetImporter.class, "line-regex", "REGEX", true, "^([^\\t]*)\\t([^\\t]*)\\t(.*)",
		 "Regular expression containing regex-groups for label, name and data.", null);

    static CommandOption.Integer nameGroup = new CommandOption.Integer
		(ExistingAlphabetImporter.class, "name", "INTEGER", true, 1,
		 "The index of the group containing the instance name.\n   Use 0 to indicate that this field is not used.", null);

    static CommandOption.Integer labelGroup = new CommandOption.Integer
		(ExistingAlphabetImporter.class, "label", "INTEGER", true, 2,
		 "The index of the group containing the label string.\n   Use 0 to indicate that this field is not used.", null);

    static CommandOption.Integer dataGroup = new CommandOption.Integer
		(ExistingAlphabetImporter.class, "data", "INTEGER", true, 3,
		 "The index of the group containing the data.", null);

    public static void writeInstanceList()
		throws IOException {

		CsvIterator reader = new CsvIterator(new FileReader(inputFile.value),
                                             lineRegex.value,
											 dataGroup.value,
											 labelGroup.value,
											 nameGroup.value);

		ArrayList<Pipe> pipes = new ArrayList<Pipe>();
		Alphabet alphabet = null;
		if (alphabetFile.value != null) {
			alphabet = AlphabetFactory.loadFromFile(alphabetFile.value);
		}
		else {
			alphabet = new Alphabet();
		}
		
		pipes.add(new CharSequence2TokenSequence(Pattern.compile(tokenRegex.value)));
		if (stoplistFile.value != null) {
			pipes.add(new TokenSequenceRemoveStopwords(stoplistFile.value, "UTF-8", false, false, false));
		}
		pipes.add(new TokenSequence2FeatureSequence(alphabet));

		//pipes.add(new PrintInput());

		Pipe serialPipe = new SerialPipes(pipes);

		InstanceList instances = new InstanceList(serialPipe);
		instances.addThruPipe(reader);
		instances.save(outputFile.value);
	}

	public static void main (String[] args) throws Exception {

		// Process the command-line options
        CommandOption.setSummary (ExistingAlphabetImporter.class,
                                  "Tool for importing text with an existing vocabulary");
        CommandOption.process (ExistingAlphabetImporter.class, args);

		writeInstanceList();
	}

}
