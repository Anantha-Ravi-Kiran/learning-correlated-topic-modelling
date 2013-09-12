#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

struct triplet{
	int docID;
	string word;
	int count;
};

void string_to_lower(string & inp_str)
{
	int i=0;
	char c;
	string::iterator s_itr = inp_str.begin();
	for(;s_itr!=inp_str.end();s_itr++)
	{
		c=*s_itr;
		*s_itr = tolower(c);
	}
}
	
int main(int argc, char *argv[])
{
	if ( argc != 2 ) /* argc should be 2 for correct execution */
    {
        /* We print argv[0] assuming it is the program name */
        printf( "usage: %s filename\n", argv[0] );
        return 0;
    }
	else
	{
		/* Read input file */
		map<string,int> vocab;	
		map<int,string> doc_words;
		vector<triplet*> M_file_list;
		int vocab_count = 1, file_count = 0, N = 0;
		string sLine_files;
		ifstream infile_names;
		string inp_file(argv[1]);
		cout << "Opening file: " << inp_file << endl;
		
		/* Extracting the path of the input file */
		unsigned found = inp_file.find_last_of('/');	
		string file_path(inp_file.substr(0,found+1)); 
		
		infile_names.open(argv[1]);
		while (!infile_names.eof())
		{
			getline(infile_names, sLine_files);
			if(sLine_files.length() < 2)
			{
				break;	
			}

			sLine_files = sLine_files;
			map<string,pair<string,int> > word_doc;
			map<string,pair<string,int> >::iterator M_d_itr;
			ifstream infile;
			infile.open(sLine_files.c_str());
			
			string sLine;
			file_count++;

			while (!infile.eof())
			{
				infile >> sLine;
				std::size_t found;
				found = sLine.find('.');
				if(found != std::string::npos)
					sLine.erase(found);
				if(sLine.empty())
					continue;
				else
				{
					int word_id;
					string_to_lower(sLine);
					pair<int,int> word_id_count;
					map<string,int>::iterator m_itr = vocab.find(sLine); 
					if(m_itr == vocab.end())
					{
						word_id = vocab_count;
						vocab[sLine] = vocab_count;
						vocab_count++;
					}	 
					
					M_d_itr = word_doc.find(sLine);
					if(M_d_itr != word_doc.end())
						word_doc[sLine].second++;
					else
					{
						word_doc[sLine].first = sLine;
						word_doc[sLine].second = 1;	
					}
				}
				N++;
			}

			infile.close();
			
			/* Creating the M_file list */
			for(M_d_itr = word_doc.begin(); M_d_itr != word_doc.end(); M_d_itr++)
			{
				pair<string,int> local_pair = M_d_itr->second;
				struct triplet *local = new triplet();
				local->docID = file_count;
				local->word = local_pair.first;
				local->count = local_pair.second;
				M_file_list.push_back(local);
			}
			
		}	
		vocab_count--;
		map<string,int>::iterator v_itr;
		int i=1;	
		for(v_itr = vocab.begin();v_itr != vocab.end(); v_itr++,i++)
		{
			v_itr->second = i;	
		}

		/* M File */
		string file_path_temp(file_path);
		file_path_temp.append("M_file.mat");
		ofstream out_file(file_path_temp.c_str());
		out_file.clear();
		out_file << file_count << endl;
		out_file << vocab_count << endl;
		out_file << N << endl;
		vector<triplet*>::iterator itr;
		for(itr = M_file_list.begin(); itr != M_file_list.end(); itr++)
		{
			struct triplet *local = *itr; 
			int wordID = vocab[local->word];
			out_file << local->docID << " " << wordID << " " << local->count << endl;
		}
		out_file.close();

		/* Vocab File */
		file_path.append("vocab.txt");
		ofstream out_file_vocab(file_path.c_str());
		out_file_vocab.clear();
		for(v_itr = vocab.begin();v_itr != vocab.end(); v_itr++)
		{
			out_file_vocab << v_itr->first << endl;	
		}
		out_file_vocab.close();

		for(itr = M_file_list.begin(); itr != M_file_list.end(); itr++)
		{
			free(*itr);	
		}

		infile_names.close();
		cout << "Read file completed!! M_nips_abstract.mat -> Output File" << endl;
	}
	
	return 0;
	
}
